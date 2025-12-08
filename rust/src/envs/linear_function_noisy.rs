// -*- coding: utf-8 -*-
/* 
(C) Copyright 2025 IBM. All Rights Reserved.

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
*/

use pyo3::prelude::*;

use rand::distributions::{Distribution, Uniform};

use twisterl::rl::env::Env;
use twisterl::python_interface::env::{PyBaseEnv, get_env_ref, get_env_mut};
use std::collections::HashMap;


use crate::envs::common::Gate;

// Define some internal representation
#[derive(Clone)]
pub struct LFState {
    pub data: Vec<bool>,
    pub map: HashMap<(usize, usize), f32>,
    size: usize,
}

// Here are some functions to manipulate the internal representation
impl LFState {
    // Constructor to create a new LinearFunction
    fn new(size: usize) -> Self {
        let mut lf = LFState {
            data: vec![false; size * size],
            size,
            map: HashMap::new(),
        };
        for i in 0..size {
            lf.set(i, i, true);
        }
        for i in 0..size {
            for j in 1..size {
                lf.insert(i, j, 0.0);

            }
        }
        lf
    }

    // Method to set a value in the LinearFunction
    fn set(&mut self, row: usize, column: usize, value: bool) {
        let index = self.index(row, column);
        self.data[index] = value;
    }

    // Method to get a value from the LinearFunction
    fn get(&self, row: usize, column: usize) -> bool {
        let index = self.index(row, column);
        self.data[index]
    }

    fn addNoise(&mut self, row: usize, column: usize, value: f32) {
        *self.map.entry((row, column)).or_insert(0.0) += value;
    }

    /// Sets (overwrites) noise value for a given edge
    fn setNoise(&mut self, row: usize, column: usize, value: f32) {
        self.map.insert((row, column), value);
    }

    /// Gets the noise value for a given edge, returning 0.0 if not present
    fn getNoise(&self, row: usize, column: usize) -> f32 {
        *self.map.get(&(row, column)).unwrap_or(&0.0)
    }

    /// Inserts a value and returns whether the key was newly inserted
    fn insert(&mut self, row: usize, column: usize, value: f32) -> bool {
        self.map.insert((row, column), value).is_none()
    }
    // Method to perform cx between q1 and q2
    fn cx(&mut self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        for column in 0..self.size {
            let a_val = self.get(q1, column);
            let b_val = self.get(q2, column);
            self.set(q2, column, a_val ^ b_val);
        }
    }

    fn swap(&mut self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        for column in 0..self.size {
            let a_val = self.get(q1, column);
            let b_val = self.get(q2, column);
            self.set(q1, column, b_val);
            self.set(q2, column, a_val);
        }
    }

    // Private helper method to calculate the linear index from row and column
    fn index(&self, row: usize, column: usize) -> usize {
        row * self.size + column
    }

    // Check if it is identity
    fn solved(&self) -> bool {
        for i in 0..self.size {
            for j in 0..self.size {
                if ((i == j) && (self.get(i, j) != true)) || ((i != j) && (self.get(i, j) != false)) {
                    return false;
                }
            }
        }
        true
    }
}

// This is the Env definition
#[derive(Clone)]
pub struct LinearFunctionNoisy {
    pub lf: LFState,
    pub depth: usize,
    pub success: bool,

    pub difficulty: usize,
    pub gateset: Vec<Gate>,
    pub depth_slope: usize,
    pub max_depth: usize,
    pub recent_noise: f32
}


impl LinearFunctionNoisy {
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
    ) -> Self {
        let  mut lf = LFState::new(num_qubits);
        let success = lf.solved();
        // let recent_noise = 0.0;
        // lf.insert(0,1,-999999999999999999.0);
        // lf.insert(1,0,-999999999999999999.0);
        // lf.insert(0,2,-999999999999999999.0);
        // lf.insert(2,0,-999999999999999999.0);
        let recent_noise = 0.0;
        LinearFunctionNoisy {lf, depth:1, success, difficulty, gateset, depth_slope, max_depth, recent_noise}
    }
    pub fn solved(&self) -> bool {
        self.lf.solved()
    }
    pub fn addNoise(&mut self, row: usize, column: usize, value: f32) {
        self.lf.addNoise(row, column, value);
    }


}

// This implements the necessary functions for the environment
impl Env for LinearFunctionNoisy {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn num_actions(&self) -> usize {
        self.gateset.len()
    }

    fn obs_shape(&self) -> Vec<usize> {
        vec![self.lf.size, self.lf.size]
    }

    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = difficulty;
    }

    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, state: Vec<i64>) {
        self.lf.data = state.iter().map(|&x| x>0).collect();
        self.depth = self.max_depth;
        self.success = self.solved();
    }

    fn reset(&mut self) {
        // Create an identity matrix for the initial 'lf' state
        self.lf = LFState::new(self.lf.size);
        self.depth = self.max_depth;
        self.success = self.solved();

        let mut rng = rand::thread_rng();
        let action_range = Uniform::new(0, self.num_actions());

        // Apply random actions based on the difficulty
        for _ in 0..self.difficulty {
            let action = action_range.sample(&mut rng);
            self.step(action);
        }
        self.depth = (self.depth_slope * self.difficulty).min(self.max_depth);
        self.success = self.solved();
    }

    fn step(&mut self, action: usize)  {
        match self.gateset[action] {
            Gate::CX(q1, q2) => {
                let mut noise = 0.0;
                if self.depth == 0 {
                    noise = -0.05;
                } else {
                    noise = -0.05;
                }
                self.lf.cx(q1, q2);
                self.lf.addNoise(q1,q2,noise);
                self.recent_noise = self.lf.getNoise(q1,q2);
                
            }
            Gate::SWAP(q1, q2) => {
                let mut noise2 = 0.0;
                if self.depth == 0 {
                    noise2 = -0.05;
                } else {
                    noise2 = -0.05;
                }
                noise2 *= 3.0;

                self.lf.swap(q1, q2);
                self.lf.addNoise(q1, q2, noise2);
                self.recent_noise = self.lf.getNoise(q1,q2);
                
            }
            _ => {}
        }        
        self.depth = self.depth.saturating_sub(1); // Prevent underflow
        self.success = self.solved();
    }
    
    fn masks(&self) -> Vec<bool> {
        vec![!self.success; self.num_actions()]
    }

    fn is_final(&self) -> bool {
        self.depth == 0 || self.success
    }

    fn reward(&self) -> f32 {
        if self.success {
            1.0 
        } else {
            if self.depth == 0 { -0.5 + self.recent_noise} else { ((-0.5  + self.recent_noise )/self.max_depth as f32)}
        }
    }

    fn observe(&self,) -> Vec<usize> {
        self.lf.data.iter()
        .enumerate() // Iterate over the Vec with indices
        .filter_map(|(index, &value)| if value { Some(index) } else { None }) // Collect indices where the value is true
        .collect()  
    }

    // fn obs2(&self,) -> Vec<usize> {
    //     self.lf.map.iter()
    //     .filter(|(_, &value)| value != 0.0)
    //     .map(|(&(r, c), _)| r * 1000 + c)   // or any packing you prefer
    //     .collect()
    // }
}


#[pyclass(name="LinearFunctionNoisyEnv", extends=PyBaseEnv)]
pub struct PyLinearFunctionNoisyEnv;

#[pymethods]
impl PyLinearFunctionNoisyEnv {
    #[new]
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize
    ) -> (Self, PyBaseEnv) {
        let env = LinearFunctionNoisy::new(num_qubits, difficulty, gateset, depth_slope, max_depth);
        let env = Box::new(env);
        (PyLinearFunctionNoisyEnv, PyBaseEnv { env })
    }
}
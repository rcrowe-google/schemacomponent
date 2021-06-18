# Lint as: python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for TFX Schema Curation Custom Executor."""

import os
import tempfile
import executor
import tensorflow as tf
from tfx import types
from tfx.utils import path_utils
from tfx.dsl.io import fileio
from tfx.types import standard_artifacts




class ExecutorTest(tf.test.TestCase):

    def _get_output_data_dir(self, sub_dir=None):
        test_dir = self._testMethodName
        if sub_dir is not None:
          test_dir = os.path.join(test_dir, sub_dir)
        return os.path.join(
           os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
           test_dir)

    def _make_base_do_params(self, source_data_dir, output_data_dir):
    # Create input dict.
        train_artifact = standard_artifacts.Examples(split='train')
        train_artifact.uri = os.path.join(source_data_dir, 'csv_example_gen/train/')
        eval_artifact = standard_artifacts.Examples(split='eval')
        eval_artifact.uri = os.path.join(source_data_dir, 'csv_example_gen/eval/')
        schema_artifact = standard_artifacts.Schema()
        schema_artifact.uri = os.path.join(source_data_dir, 'schema_gen/')

        self._input_dict = {
            'input_data': [train_artifact, eval_artifact],
            'schema': [schema_artifact],
        }

        self.output_data_train = standard_artifacts.Examples(
        split='train')

        self.output_data_train.uri = os.path.join(output_data_dir,
                                                        'train')

        self.output_data_eval = standard_artifacts.Examples(
        split='eval')

        self._output_data_eval.uri = os.path.join(output_data_dir, 'eval')
        
        temp_path_output = types.Artifact('TempPath')
        temp_path_output.uri = tempfile.mkdtemp()

        self.output_dict = {
           'output_data': [
            self.output_data_train, self.output_data_eval
            ],
            'temp_path': [temp_path_output],
        }

        self._exec_properties = {}

        self._executor = executor.Executor()

    def _verify_model_exports(self):
        self.assertTrue(
            
            fileio.exists(path_utils.eval_model_dir(self.output_data_eval_uri)))
        self.assertTrue(
        fileio.exists(path_utils.serving_model_dir(self.output_data_train.uri )))

    def _verify_no_eval_model_exports(self):
        self.assertFalse(
         fileio.exists(path_utils.eval_model_dir(self.output_data_eval.uri )))

    def _verify_model_run_exports(self):
        self.assertTrue(fileio.exists(os.path.dirname(self.tput_data_train.uri )))

    def _do(self, test_executor):
        test_executor.Do(
          input_dict=self._input_dict,
          output_dict=self._output_dict,
          exec_properties=self._exec_properties)
         



if __name__ == '__main__':
  tf.test.main()



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
"""Tests for tfx.components.custom.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import tensorflow as tf
from tfx import types
import executor
from tfx.types import standard_artifacts
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.components.util import udf_utils
from tfx.types import standard_component_specs
from tfx.components.testdata.module_file import transform_module
import tensorflow_transform as tft



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
    schema_artifact = standard_artifacts.Schema()
    schema_artifact.uri = os.path.join(source_data_dir, 'schema_gen/')

    self._input_dict = {
        'schema': [schema_artifact],
    }

    # Create output dict.
    self._output = standard_artifacts.TransformGraph()
    self._output.uri = os.path.join(output_data_dir,
                                                'custom_output')
    temp_path_output = types.Artifact
    temp_path_output.uri = tempfile.mkdtemp()

    self._output_dict = {
        'custom_output': [self._output],
        'temp_path': [temp_path_output],
    }

    # Create exec properties skeleton.
    self._exec_properties = {}


  def setUp(self):
    super(ExecutorTest, self).setUp()

    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = self._get_output_data_dir()

    self._make_base_do_params(self._source_data_dir, self._output_data_dir)

    # Create exec properties skeleton.
    self._module_file = os.path.join(self._source_data_dir,
                                     'module_file/custom_module.py')
    self._preprocessing_fn = '%s.%s' % (
        transform_module.preprocessing_fn.__module__,
        transform_module.preprocessing_fn.__name__)

    # Executor for test.
    self._custom_executor = executor.Executor()

  def _verify_outputs(self):
    path_to_saved_model = os.path.join(
        self._output.uri, tft.TFTransformOutput.TRANSFORM_FN_DIR,
        tf.saved_model.SAVED_MODEL_FILENAME_PB)
    self.assertTrue(tf.io.gfile.exists(path_to_saved_model))
  
  def testDoWithModuleFile(self):
    self._exec_properties['module_file'] = self._module_file
    self._custom_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_outputs()

  def testDoWithPreprocessingFn(self):
    self._exec_properties['preprocessing_fn'] = self._preprocessing_fn
    self._custom_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_outputs()

  def testDoWithNoPreprocessingFn(self):
    with self.assertRaises(ValueError):
      self._custom_executor.Do(self._input_dict, self._output_dict,
                                  self._exec_properties)

  def testDoWithDuplicatePreprocessingFn(self):
    self._exec_properties['module_file'] = self._module_file
    self._exec_properties['preprocessing_fn'] = self._preprocessing_fn
    with self.assertRaises(ValueError):
      self._custom_executor.Do(self._input_dict, self._output_dict,
                                  self._exec_properties)

  def testDoWithCache(self):
    # First run that creates cache.
    output_cache_artifact = types.Artifact('OutputCache')
    output_cache_artifact.uri = os.path.join(self._output_data_dir, 'CACHE/')

    self._output_dict['cache_output_path'] = [output_cache_artifact]

    self._exec_properties['module_file'] = self._module_file
    self._custom_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_outputs()
    self.assertNotEqual(0,
                        len(tf.io.gfile.listdir(output_cache_artifact.uri)))
    
    # Second run from cache.
    self._output_data_dir = self._get_output_data_dir('2nd_run')
    input_cache_artifact = types.Artifact('InputCache')
    input_cache_artifact.uri = output_cache_artifact.uri

    output_cache_artifact = types.Artifact('OutputCache')
    output_cache_artifact.uri = os.path.join(self._output_data_dir, 'CACHE/')

    self._make_base_do_params(self._source_data_dir, self._output_data_dir)

    self._input_dict['cache_input_path'] = [input_cache_artifact]
    self._output_dict['cache_output_path'] = [output_cache_artifact]

    self._exec_properties['module_file'] = self._module_file
    self._custom_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)

    self._verify_outputs()
    self.assertNotEqual(0,
                        len(tf.io.gfile.listdir(output_cache_artifact.uri)))






if __name__ == '__main__':
  tf.test.main()

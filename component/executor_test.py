# Lint as: python2, python3
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
"""Tests for schemaCuration.component.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text
from six import string_types

import tensorflow as tf
import executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils
import sys
sys.path.append('.')

from schemacomponent.test_data.module_file import module_file

class ExecutorTest(tf.test.TestCase):

  def testDo(self):
    super(ExecutorTest, self).setUp()
    self._source_data_dir = os.path.join('schemacomponent', 'test_data')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(self._source_data_dir, 'schema_gen')

    input_dict = {
       
        standard_component_specs.SCHEMA_KEY: [schema], 
    }

    schema_output = standard_artifacts.Schema()
    schema_output.uri = os.path.join(self._output_data_dir, 'custom_schema')

    output_dict = {
        'custom_schema': [schema_output],
    }

   
    _module_file = os.path.join(self._source_data_dir,
                                     standard_component_specs.MODULE_FILE_KEY,
                                     'module_file.py')
    schema_fn = '%s.%s' % (module_file.schema_fn.__module__,
                                  module_file.schema_fn.__name__)

    print(_module_file)
    exec_properties = {
        standard_component_specs.MODULE_FILE_KEY : _module_file
        
    }

    schemaCuration_executor = executor.Executor()

    schemaCuration_executor.Do(input_dict, output_dict, exec_properties)

    self.assertNotEqual(0, len(fileio.listdir(schema_output.uri)))
    

if __name__ == '__main__':
    tf.test.main()

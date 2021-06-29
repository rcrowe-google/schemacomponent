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
"""Tests for TFX Schema Curation Custom Component."""

import json

import tensorflow as tf

from tfx.types import artifact
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.types import artifact_utils
import component


class SchemaCurationTest(tf.test.TestCase):

  def testConstruct(self):
    schema_curation = component.SchemaCuration(
        input_schema=channel_utils.as_channel([standard_artifacts.Schema()]),
        )
    self.assertEqual(
        standard_artifacts.Schema.TYPE_NAME,
        schema_curation.outputs['output_schema'].type_name)
    

if __name__ == '__main__':
  tf.test.main()

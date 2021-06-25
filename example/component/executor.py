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
"""TFX Schema curation custom component executor"""


import json
import os
from typing import Any, Dict, List, Text
from absl import logging

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.components.util import udf_utils
from tfx.types import standard_component_specs

_DEFAULT_FILE_NAME = 'schema.pbtxt'

class Executor(base_executor.BaseExecutor):
  """Executor for TFX Schema Curation Custom Component.

  This Executor will execute the schema_fn with correct parameters by resolving the input
  artifacts, output artifacts and execution properties.
  """
  

  
  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Uses a user-supplied schema_fn to curate a schema.

    The schema curation Executor invokes a schema_fn callback function provided by
    the user via the module_file parameter. In this function, user changes the schema.

    Args:
      input_dict: Input dict from input key to a list of ML-Metadata Artifacts.including:
        - schema: Schema of the data.A list of type `standard_artifacts.Schema` which should
          contain a single schema artifact.
      output_dict: Output dict from output key to a list of Artifacts.
        - schema: Schema of the data.A list of type `standard_artifacts.Schema` which should
          contain a single schema artifact.
      exec_properties: A dict of execution properties, including:
        - module_file: The file path to a python module file, from which the
          'schema_fn' function will be loaded. Exactly one of
          'module_file', 'module_path' and 'schema_fn' should be set.
        - module_path: The python module path, from which the
          'schema_fn' function will be loaded. Exactly one of
          'module_file', 'module_path' and 'schema_fn' should be set.
        - schema_fn: The module path to a python function that
          implements 'schema_fn'. Exactly one of 'module_file',
          'module_path' and 'schema_fn' should be set.
        
    Returns:
      None

    Raises:
      ValueError: When not exactly one of `module_file`, `module_path` and
        `schema_fn` are present in 'exec_properties'.
      RuntimeError: If schema_fn failed to generate modified output schema.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    if input_dict.get(standard_component_specs.SCHEMA_KEY):
      schema_file = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(
            input_dict[standard_component_specs.SCHEMA_KEY]))
    else:
      schema_file = None

    schema = io_utils.SchemaReader().read(schema_file)
    schema_fn = udf_utils.get_fn(exec_properties, 'schema_fn')
    output_schema = schema_fn(schema)

    output_uri = os.path.join(
        artifact_utils.get_single_uri(
            output_dict[standard_component_specs.SCHEMA_KEY]),
        _DEFAULT_FILE_NAME)
    io_utils.write_pbtxt_file(output_uri, output_schema)

    logging.info('Schema written to %s.', output_uri)
    
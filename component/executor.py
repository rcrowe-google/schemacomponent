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
"""Example of a Hello World TFX custom component.

This custom component simply passes examples through. This is meant to serve as
a kind of starting point example for creating custom components.

This component along with other custom component related code will only serve as
an example and will not be supported by TFX team.
"""

import json
import os
from typing import Any, Dict, List, Text


from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.utils import io_utils

_DEFAULT_FILE_NAME = 'schema.pbtxt'

class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Copy the input_data to the output_data.

    For this example that is all that the Executor does.  For a different
    custom component, this is where the real functionality of the component
    would be included.

    This component both reads and writes Examples, but a different component
    might read and write artifacts of other types.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - input_data: A list of type `standard_artifacts.Examples` which will
          often contain two splits, 'train' and 'eval'.
      output_dict: Output dict from key to a list of artifacts, including:
        - output_data: A list of type `standard_artifacts.Examples` which will
          usually contain the same splits as input_data.
      exec_properties: A dict of execution properties, including:
        - name: Optional unique name. Necessary iff multiple Hello components
          are declared in the same pipeline.

    Returns:
      None

    Raises:
      OSError and its subclasses
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # Load and deserialize exclude splits from execution properties.
    exclude_splits = json_utils.loads(
        exec_properties.get('exclude_splits', 'null')) or []
    if not isinstance(exclude_splits, list):
      raise ValueError('exclude_splits in execution properties needs to be a '
                       'list. Got %s instead.' % type(exclude_splits))

    # Setup output splits.
    stats_artifact = artifact_utils.get_single_instance(
        input_dict['statistics'])
    stats_split_names = artifact_utils.decode_split_names(
        stats_artifact.split_names)
    split_names = [
        split for split in stats_split_names if split not in exclude_splits
    ]

    schema = io_utils.SchemaReader().read(
        io_utils.get_only_uri_in_dir(
            artifact_utils.get_single_uri(
                input_dict['schema'])))

    for split in artifact_utils.decode_split_names(stats_artifact.split_names):
      if split in exclude_splits:
        continue
      logging.info(
          'Curating schema against the computed statistics for '
          'split %s.', split)

    output_schema = schema

    output_uri = os.path.join(
        artifact_utils.get_single_uri(
            output_dict['schema']),
        _DEFAULT_FILE_NAME)
    io_utils.write_pbtxt_file(output_uri, output_schema)
    logging.info('Schema written to %s.', output_uri)
    
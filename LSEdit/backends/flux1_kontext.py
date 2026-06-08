from __future__ import annotations

from typing import Any

import torch

from havedit.backends.base import BackendRunContext


class Flux1KontextBackendAdapter:
    backend_name = "flux1-kontext"

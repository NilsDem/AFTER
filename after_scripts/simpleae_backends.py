"""Export and Python runtime adapters for the SimpleAE benchmark."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class Runtime(Protocol):
    def reset(self) -> None: ...
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


@contextlib.contextmanager
def quiet_native_output():
    """Suppress exporter and native-runtime startup diagnostics."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as null, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.dup2(null.fileno(), 1)
            os.dup2(null.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


class TorchRuntime:
    def __init__(self, model: torch.nn.Module, state: torch.Tensor):
        self.model = model
        self.initial_state = state
        self.state = state.clone()

    def reset(self) -> None:
        self.state = self.initial_state.clone()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y, self.state = self.model(torch.from_numpy(x), self.state)
        return y.numpy()


class TorchScriptRuntime:
    def __init__(self, path: Path, state: torch.Tensor):
        self.model = torch.jit.load(str(path)).eval()
        self.initial_state = state
        self.state = state.clone()

    def reset(self) -> None:
        self.state = self.initial_state.clone()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            y, self.state = self.model(torch.from_numpy(x), self.state)
        return y.numpy()


class OnnxRuntime:
    def __init__(
        self,
        path: Path,
        state: torch.Tensor,
        threads: int,
        input_name: str,
        output_name: str,
    ):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.initial_state = state.numpy()
        self.state = self.initial_state.copy()
        self.input_name = input_name
        self.output_name = output_name

    def reset(self) -> None:
        self.state = self.initial_state.copy()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y, self.state = self.session.run(
            [self.output_name, "state_out"],
            {self.input_name: x, "cache": self.state},
        )
        return y


class CoreMLRuntime:
    def __init__(
        self,
        path: Path,
        state: torch.Tensor,
        input_name: str,
        output_name: str,
    ):
        import coremltools as ct
        compiled_path = path.with_suffix(".mlmodelc")
        # self.model = ct.models.MLModel(str(path))
        self.model = ct.models.CompiledMLModel(
            str(compiled_path),
            compute_units=ct.ComputeUnit.ALL,
        )
        self.initial_state = state.numpy()
        self.state = self.initial_state.copy()
        self.input_name = input_name
        self.output_name = output_name

    def reset(self) -> None:
        self.state = self.initial_state.copy()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        result = self.model.predict({self.input_name: x, "cache": self.state})
        self.state = result["state_out"]
        return result[self.output_name]


class CoreMLStatefulRuntime:
    def __init__(self, path: Path, input_name: str, output_name: str):
        import coremltools as ct

        # self.model = ct.models.MLModel(str(path))
        compiled_path = path.with_suffix(".mlmodelc")
                # self.model = ct.models.MLModel(str(path))
        self.model = ct.models.CompiledMLModel(
                    str(compiled_path),
                    compute_units=ct.ComputeUnit.ALL,
                )

        
        self.state = self.model.make_state()
        self.input_name = input_name
        self.output_name = output_name

    def reset(self) -> None:
        self.state = self.model.make_state()

    def __call__(self, x: np.array) -> np.array:
        result = self.model.predict({self.input_name: x}, state=self.state)
        return result[self.output_name]


class ExecuTorchRuntime:
    def __init__(self, path: Path, state: torch.Tensor):
        from executorch.runtime import Runtime as ETRuntime
        from executorch.runtime import Verification

        program = ETRuntime.get().load_program(
            path, verification=Verification.Minimal
        )
        self.program = program
        self.method = program.load_method("forward")
        self.initial_state = state
        self.state = state.clone()

    def reset(self) -> None:
        self.state = self.initial_state.clone()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y, self.state = self.method.execute((torch.from_numpy(x), self.state))
        y=y.numpy()
        return y


class AOTIRuntime:
    def __init__(self, path: Path, state: torch.Tensor):
        self.model = torch._inductor.aoti_load_package(
            str(path),
            run_single_threaded=True,
        )

        self.initial_state = state.clone()
        self.state = state.clone()

    def reset(self) -> None:
        self.state = self.initial_state.clone()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # zero-copy on CPU
        x = torch.from_numpy(x)

        y, self.state = self.model(
            x,
            self.state,
        )

        return y.numpy()


class AOTIProfileRuntime(AOTIRuntime):
    def __init__(self, path: Path, state: torch.Tensor):
        super().__init__(path, state)
        self.trace_path = path.with_name(f"{path.stem}_trace.json")

    def profile(self, x: np.ndarray, warmup: int = 20, reps: int = 100) -> None:
        x = torch.from_numpy(x)
        self.reset()
        with torch.inference_mode():
            for _ in range(warmup):
                _, self.state = self.model(x, self.state)

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
            ) as prof:
                for _ in range(reps):
                    _, self.state = self.model(x, self.state)

        print(prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=50,
        ))
        prof.export_chrome_trace(str(self.trace_path))
        print(f"AOTI Chrome trace: {self.trace_path}")
        self.reset()


def artifact_suffix(backend: str) -> str:
    return {
        "torchscript": ".pt",
        "onnx": ".onnx",
        "coreml": ".mlpackage",
        "coreml-stateful": ".mlpackage",
        "xnnpack": ".pte",
        "mlx": ".pte",
        "aoti": ".pt2",
        "aoti-profile": ".pt2",
    }[backend]


def export_torchscript(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
    path: Path,
) -> None:
    exported = torch.export.export(model, inputs, strict=False)
    traced = torch.jit.trace(
        exported.module(),
        inputs,
        strict=False,
        check_trace=False,
    )
    torch.jit.save(traced, str(path))


def export_onnx(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
    path: Path,
) -> None:
    torch.onnx.export(
        model,
        inputs,
        str(path),
        input_names=[model.input_name, "cache"],
        output_names=[model.output_name, "state_out"],
        opset_version=18,
        dynamo=True,
    )


def export_coreml(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
    path: Path,
) -> None:
    import coremltools as ct

    exported = torch.export.export(model, inputs, strict=False).run_decompositions({})
    graph = exported.graph
    for node in list(graph.nodes):
        if node.target == torch.ops.aten.alias.default:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
    graph.lint()
    exported.graph_module.recompile()
    converted = ct.convert(
        exported,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name=model.input_name, shape=tuple(inputs[0].shape)),
            ct.TensorType(name="cache", shape=tuple(inputs[1].shape)),
        ],
        outputs=[
            ct.TensorType(name=model.output_name),
            ct.TensorType(name="state_out"),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    converted.save(str(path))
    compiled_path = path.with_suffix(".mlmodelc")
    if compiled_path.is_dir():
        shutil.rmtree(compiled_path)
    elif compiled_path.exists():
        compiled_path.unlink()
    _ = ct.models.utils.compile_model(
        str(path),
        destination_path=str(compiled_path),
    )


def export_coreml_stateful(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
    path: Path,
) -> None:
    import coremltools as ct
    import numpy as np

    from after.autoencoder.networks.RofNet import StatelessStreamingRofNet
    from after_scripts.simpleae_export_model import (
        CoreMLStatefulExport,
        CoreMLStatefulSimpleAE,
    )

    if isinstance(model, StatelessStreamingRofNet):
        stateful = CoreMLStatefulExport(model).eval()
    else:
        stateful = CoreMLStatefulSimpleAE(model).eval()
    exported = torch.export.export(stateful, (inputs[0],), strict=False)
    exported = exported.run_decompositions({})
    graph = exported.graph
    for node in list(graph.nodes):
        if node.target == torch.ops.aten.alias.default:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
    graph.lint()
    exported.graph_module.recompile()
    converted = ct.convert(
        exported,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=model.input_name, shape=tuple(inputs[0].shape))],
        outputs=[ct.TensorType(name=model.output_name)],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=shape,
                    dtype=np.float16,
                ),
                name=name,
            )
            for name, shape in zip(stateful.state_names, model.cache_shapes)
        ],
        minimum_deployment_target=ct.target.macOS15,
    )
    converted.save(str(path))
    compiled_path = path.with_suffix(".mlmodelc")
    if compiled_path.is_dir():
        shutil.rmtree(compiled_path)
    elif compiled_path.exists():
        compiled_path.unlink()
    _ = ct.models.utils.compile_model(
        str(path),
        destination_path=str(compiled_path),
    )


def export_executorch(
    backend: str,
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
    path: Path,
) -> None:
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        partitioner = XnnpackPartitioner()
    elif backend == "mlx":
        from executorch.backends.mlx.partitioner import MLXPartitioner

        partitioner = MLXPartitioner()
    else:
        raise ValueError(backend)

    attention_model = getattr(model, "model", None)
    original_attention_impl = getattr(attention_model, "attention_impl", None)
    if original_attention_impl is not None:
        attention_model.set_attention_impl("manual")
    try:
        exported = torch.export.export(model, inputs, strict=True)
        edge = to_edge_transform_and_lower(
            exported,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[partitioner],
        )
        edge.to_executorch().save(str(path))
    finally:
        if original_attention_impl is not None:
            attention_model.set_attention_impl(original_attention_impl)


def export_aoti(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
    path: Path,
    profile: bool = False,
) -> None:
    model = model.eval()

    exported = torch.export.export(
        model,
        inputs,
        strict=True,
    )

    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(path),
        inductor_configs={
            "max_autotune": True,
            # Some macOS/PyTorch configurations expose no valid C++ choice for
            # small Roformer matrix multiplies. Keep ATen as the guaranteed
            # fallback while still allowing the tuned C++ kernels.
            "max_autotune_gemm_backends": "ATEN,CPP",
            "cpp.enable_kernel_profile": profile and sys.platform in ("linux", "win32"),
            "profiler_mark_wrapper_call": profile,
        },
    )

def export_backend(
    backend: str,
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    with quiet_native_output(), contextlib.redirect_stdout(
        io.StringIO()
    ), contextlib.redirect_stderr(io.StringIO()):
        if backend == "torchscript":
            export_torchscript(model, inputs, path)
        elif backend == "onnx":
            export_onnx(model, inputs, path)
        elif backend == "coreml":
            export_coreml(model, inputs, path)
        elif backend == "coreml-stateful":
            export_coreml_stateful(model, inputs, path)
        elif backend in ("xnnpack", "mlx"):
            export_executorch(backend, model, inputs, path)
        elif backend in ("aoti", "aoti-profile"):
            export_aoti(model, inputs, path, profile=backend == "aoti-profile")
        else:
            raise ValueError(f"Unknown export backend: {backend}")


def load_backend(
    backend: str,
    path: Path,
    model: torch.nn.Module,
    state: torch.Tensor,
    threads: int,
) -> Runtime:
    if backend == "torch":
        return TorchRuntime(model, state)
    with quiet_native_output():
        if backend == "torchscript":
            return TorchScriptRuntime(path, state)
        if backend == "onnx":
            return OnnxRuntime(
                path, state, threads, model.input_name, model.output_name
            )
        if backend == "coreml":
            return CoreMLRuntime(path, state, model.input_name, model.output_name)
        if backend == "coreml-stateful":
            return CoreMLStatefulRuntime(
                path, model.input_name, model.output_name
            )
        if backend == "aoti-profile":
            return AOTIProfileRuntime(path, state)
        if backend == "aoti":
            return AOTIRuntime(path, state)
        if backend in ("xnnpack", "mlx"):
            return ExecuTorchRuntime(path, state)
    raise ValueError(f"Unknown runtime backend: {backend}")

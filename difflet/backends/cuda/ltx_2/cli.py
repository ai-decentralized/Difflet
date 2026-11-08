"""LTX-2 request differences on the existing shared GPU CLI lifecycle."""

from types import SimpleNamespace

from difflet.backends.gpu.cli import GpuOrchestrator
from difflet.cli.dp import stage_loop

from .policy import positive_integer, validate_request


class CudaLtx2Orchestrator(GpuOrchestrator):
    def refuse_unsupported(self):
        args = self.args
        try:
            validate_request(
                model_id=args.model_id,
                shape={
                    name: getattr(args, name, None) for name in ("height", "width", "num_frames")
                },
                parallel=SimpleNamespace(
                    tp_degree=getattr(args, "tp_degree", None),
                    cp_degree=getattr(args, "cp_degree", None),
                    dp_degree=getattr(args, "dp", None),
                    sp_enabled=getattr(args, "sp_enabled", False),
                    cfg_parallel_enabled=getattr(args, "cfg_parallel", False),
                ),
            )
            if args.steps is not None:
                positive_integer(args.steps, "steps")
        except (ValueError, NotImplementedError) as exc:
            raise SystemExit(f"{self.tag} {exc}") from None
        super().refuse_unsupported()

    def generate_kwargs(self, request=None):
        kwargs = super().generate_kwargs(request)
        negative = stage_loop.effective(request, self.args, "negative_prompt", None)
        if negative is not None:
            kwargs["negative_prompt"] = negative
        return kwargs

    def download(self):
        self.refuse_unsupported()
        from difflet.pipeline.path_resolver import resolve_model_path
        from difflet.registry import resolve_model

        resolve_model_path(
            self.model_id,
            revision=self.args.revision,
            local_files_only=False,
            allow_patterns=resolve_model(self.model_id).download_patterns,
        )
        print(f"[difflet] weights ready for {self.model_id}")

#!/usr/bin/env python3
"""Convert the checkpoint's trunk to GGUF, for llama-server and whatever else reads GGUF.

The checkpoint is a `Qwen3_5ForSequenceClassification`: the Qwen3.5 trunk cut after block 27
plus a computed head. llama.cpp's converter knows the trunk (`Qwen3_5ForCausalLM`, arch
`qwen35`) but not the classifier name, and it would refuse the head tensor as unmapped. So the
class is registered under the classifier's name here and `score.weight` is dropped: the head is
read on the client from the 195 KB shard on the Hub, the GGUF carries the embedder only.

    LLAMA_CPP=~/Projects/llama.cpp scripts/convert_gguf.py <checkpoint dir> <out.gguf> [bf16|q8_0]

`LLAMA_CPP` is a checkout of llama.cpp (the converter is a script there, not a package); a Q8_0
file comes from `llama-quantize <bf16.gguf> <q8_0.gguf> Q8_0` on the bf16 one. Everything else — tokenizer, `layer_types`, tied embeddings — is what the stock converter does.
"""
import os
import sys
from pathlib import Path

LLAMA = Path(os.environ.get("LLAMA_CPP", Path.home() / "Projects" / "llama.cpp"))
sys.path.insert(0, str(LLAMA))
sys.path.insert(0, str(LLAMA / "gguf-py"))

import convert_hf_to_gguf as convert  # noqa: E402
import conversion  # noqa: E402
from conversion import ModelBase  # noqa: E402
from conversion.qwen import Qwen3_5TextModel  # noqa: E402
import gguf  # noqa: E402


@ModelBase.register("Qwen3_5ForSequenceClassification")
class TypecastTrunk(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # The file says how it is to be read: last-token pooling, so a launcher that honours
        # the key (llama-server does) needs no `--pooling last` and cannot average by mistake.
        self.gguf_writer.add_pooling_type(gguf.PoolingType.LAST)

    def modify_tensors(self, data_torch, name, bid):
        if name == "score.weight":
            print(f"dropping {name} {tuple(data_torch.shape)}: the head is read on the client",
                  flush=True)
            return []
        return super().modify_tensors(data_torch, name, bid)


# The converter resolves an architecture through a static name -> module map before it looks
# at the registry, so the classifier's name has to be in both.
conversion.TEXT_MODEL_MAP["Qwen3_5ForSequenceClassification"] = "qwen"


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    outtype = sys.argv[3] if len(sys.argv) > 3 else "bf16"
    # `--no-mtp`: the converter would otherwise count Qwen3.5's next-token-prediction block
    # into block_count, and this checkpoint carries no such block (28 in the file, 29 declared,
    # and the server refuses the file for the missing blk.28).
    sys.argv = ["convert_hf_to_gguf.py", str(src), "--outfile", str(out), "--outtype", outtype,
                "--no-mtp"]
    convert.main()


if __name__ == "__main__":
    main()

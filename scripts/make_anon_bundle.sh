#!/usr/bin/env bash
# 生成双盲投稿用匿名代码包(supplementary material)。
# 用法: bash scripts/make_anon_bundle.sh [输出zip名]
# 产物: /tmp/apc-anon-bundle.zip — 可上传 OpenReview/EMNLP 辅助材料。
# 策略: git archive(dev HEAD) -> 删身份文件 -> 重写 README 版权行 -> 排除内部文献笔记
#        -> 校验零身份残留 -> 打包。
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-/tmp/apc-anon-bundle.zip}"
STAGE="$(mktemp -d /tmp/apc-anon-XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

git archive --format=tar HEAD | tar -x -C "$STAGE"

# 1) 身份/内部文件:LICENSE 署名、内部文献管理笔记(含 pre-reg 讨论,非复现必需)
rm -f "$STAGE/LICENSE" "$STAGE/docs/literature/frontier-2026.md"

# 2) README:去署名与仓库链接,保留全部技术内容
sed -i 's|\[MIT\](LICENSE) © 2026 lanshi17|MIT — license text withheld for double-blind review; restored in camera-ready.|' "$STAGE/README.md"

# 3) 全树校验:任何 lanshi / 个人路径 / 仓库 URL 残留 = 致命,拒绝出包
if grep -rIl --exclude-dir=.git -e 'lanshi17' -e 'lanshi' -e '/mnt/data/Projects' "$STAGE" | grep .; then
  echo "ABORT: identity leak above" >&2
  exit 1
fi
if grep -rIlE '\bsk-[A-Za-z0-9]{16,}\b|\bDASHSCOPE_API_KEY=[A-Za-z0-9]{8,}' "$STAGE" | grep .; then
  echo "ABORT: credential-pattern leak" >&2
  exit 1
fi

# 4) 匿名声明置顶
{ printf '%s\n' "> Anonymized review bundle. Build: scripts/make_anon_bundle.sh; deterministic from git HEAD." "" ; cat "$STAGE/README.md"; } > "$STAGE/README.tmp" && mv "$STAGE/README.tmp" "$STAGE/README.md"

(cd "$STAGE" && zip -qr "$OUT" .)
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1)) — leak scan clean, $(find "$STAGE" -type f | wc -l) files"

# docs/assets/demo.tape — the README demo, recorded with one command from the repo root:
#
#   vhs docs/assets/demo.tape
#
# Precondition: at least one run must exist in the workspace, or the board records
# empty and `run-record latest` has nothing to resolve. Record after a real session:
#
#   uv run python -m noctis run -v --time-limit-hours 1 \
#     --directive "find a mean-reversion strategy on very volatile stocks; high risk appetite"
#
# Budget (ASSETS.md#terminal-demos): the committed recording stays under 8 MB and
# under 12 s. Length is arithmetic, not a guess —
#
#   sleeps:  500ms + 1500ms + 2000ms + 2500ms          = 6.5s
#   typing:  (20 + 18 + 49 chars) x 40ms TypingSpeed   ≈ 3.5s
#   total                                              ≈ 10.0s, 2s under the ceiling
#
# Over budget after recording? Shrink the canvas before you shorten the demo:
# Width, Height and FontSize are the three dials that decide the file size.
#
# This recording spends the README's whole animation budget: at most 1 animated
# element per README, and it is the demo, never the hero.

Output docs/assets/demo.webp

# Determinism. The shell and the prompt are pinned so two machines record the same
# frames; without this the recording carries whatever PS1 the operator happens to
# have. The setup keystrokes are hidden, so the demo opens on a clean prompt.
Set Shell "bash"
Set Theme "Catppuccin Mocha"

Hide
Type "export PS1='$ ' && clear"
Enter
Show

# The canvas. Explicit, because a recording sized by the recorder's defaults looks
# different on the next machine. Dark and opaque, so it reads in both GitHub themes.
Set FontSize 22
Set Width 1200
Set Height 600
Set Padding 20
Set TypingSpeed 40ms

# The script: set the scene, show the work, prove the result. The slow command gets
# the long Sleep — that pause is the reader reading its output.
Sleep 500ms

Type "uv run noctis status"
Enter
Sleep 1500ms

Type "uv run noctis runs"
Enter
Sleep 2s

Type "uv run noctis run-record latest | jq .performance"
Enter
Sleep 2500ms

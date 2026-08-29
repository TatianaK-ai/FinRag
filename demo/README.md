# Demo video

**Watch it here — this one plays in the browser:**

https://github.com/user-attachments/assets/eac1c2bd-ba70-4d93-9eb6-576f6b84a639

Also playable on [issue #1](https://github.com/TatianaK-ai/FinRag/issues/1).

3:37, recorded against the running server:

| | |
| --- | --- |
| 0:00 | What the system is, and why refusing is the hard part |
| 0:22 | *"What were Apple's total net sales in fiscal 2025?"* → $416,161 million with six EDGAR citations |
| 1:02 | *"How much did Tesla spend on R&D?"* → refused, nothing in the corpus matched |
| | *"What price did NVIDIA pay for Intel's foundry business?"* → refused; the answer cites no passage mentioning Intel |
| 1:44 | *"What was revenue last year?"* → asks back, with the graph trace expanded |
| 2:10 | The LangGraph pipeline and the four refusal gates |
| 2:52 | Evaluation results, and what they do not prove |
| 3:22 | The scope-gate regression that cost 75 points of correct refusal |

## About `finrag-demo-live.mp4`

The same video is committed here as [`finrag-demo-live.mp4`](finrag-demo-live.mp4)
so the repository is self-contained. **That link downloads rather than plays** —
GitHub serves repository files as `application/octet-stream`, and its blob viewer
does not render a video player for files in a repo at any size. Use one of the
two links above to watch it.

Audio is a synthesised voice (Windows SAPI), not a human recording.

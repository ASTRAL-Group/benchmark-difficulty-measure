# Third-Party Notices

Benchmark items in `data/` come from the sources below and remain under their original terms. Source licenses apply to their own content, not automatically to the rest of this repository. Release transformations are serialization, a compact `item_id` key with source identifiers preserved, numeric problem ordering for AIME24/AMC23, and the ScienceQA image representation noted below. No question or answer was rewritten.

## Sources

- **AMC23 and AIME24/25/26:** author-processed evaluation snapshots sourced from `evaluate/evaluation/data/<source-stem>/test.jsonl`, preserving all original questions, answers, and metadata. Original source IDs are retained in `url` for AMC23/AIME24 and `id` for AIME25/AIME26 (strings `0`–`29`, not inferred competition problem numbers). Release files join on the added integer `item_id`; AMC23/AIME24 are naturally ordered by session and problem number. These are Mathematical Association of America competition materials and retain their source terms; local inclusion does not establish a new redistribution license. The [MAA educational-sharing notice](https://maa.org/wp-content/uploads/2024/08/2022-AMC-10A-solutions.pdf) describes educational sharing and permissions for revenue-generating reuse.
- **GPQA-Diamond:** Rein et al., [Idavidrein/gpqa revision](https://huggingface.co/datasets/Idavidrein/gpqa/blob/633f5ee89ab8ad4522a9f850766b73f62147ffdd/README.md) `633f5ee89ab8ad4522a9f850766b73f62147ffdd`, `gpqa_diamond`, `train`, 198 items. GPQA is released under CC BY 4.0 with access terms prohibiting public disclosure of examples as plaintext or images, so this release contains no GPQA questions or answers. `test.jsonl` holds only `item_id` and `source_item_id`, the original `Record ID`, recovered by uniquely matching each evaluated question against that revision.
- **MATH-500:** [OpenAI PRM800K test split](https://github.com/openai/prm800k/blob/7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/math_splits/test.jsonl), MIT; all 500 source objects match the evaluated inputs. The original `unique_id` is retained; release files join on `item_id`.
- **Minerva-Math:** the evaluated 272-item MIT OpenCourseWare collection from Lewkowycz et al., [Solving Quantitative Reasoning Problems with Language Models](https://openreview.net/forum?id=IFXTZERXdM7). Original `idx`, problem, solution and type are preserved. [MIT OCW terms](https://ocw.mit.edu/pages/privacy-and-terms-of-use/) require [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): attribution, noncommercial use and share-alike. Credit MIT OpenCourseWare and the original instructors.
- **OlympiadBench:** He et al., [OpenBMB/OlympiadBench](https://github.com/OpenBMB/OlympiadBench), evaluated 675-item English mathematics subset, original `id`. The [upstream repository](https://github.com/OpenBMB/OlympiadBench/blob/main/LICENSE) is MIT; the [dataset card](https://huggingface.co/datasets/Hothan/OlympiadBench) labels the dataset Apache-2.0. Both notices are retained below.
- **Gaokao2023en:** [the exact Qwen2.5-Math evaluation input](https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/data/gaokao2023en/test.jsonl), distributed with the [MIT evaluation license](https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/LICENSE), copyright Zhibin Gou. All 385 original objects match this source; `qN` identifies its zero-based row. Preserve the source's existing mixed `source` labels; no subset was substituted.
- **MMLU-Pro:** Wang et al., [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/54611cde22c74cca43dd78732198de6abe971398), revision `54611cde22c74cca43dd78732198de6abe971398`, `test`, filtered by category without reordering. The pinned dataset card declares MIT; the [project license](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/LICENSE) is Apache-2.0. Biology retains the original evaluation JSONL's `qN` IDs; chemistry and physics retain source `question_id` values. Biology's questions, options and answers match the corresponding source category exactly.
- **BBH logical deduction:** Suzgun et al., [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard), MIT, copyright 2022 suzgunmirac. Evaluated [SaylorTwift/bbh revision](https://huggingface.co/datasets/SaylorTwift/bbh/tree/b5306be6f827cfafbb545ff5a51f96916029b0fd) `b5306be6f827cfafbb545ff5a51f96916029b0fd`, configs `logical_deduction_three_objects`, `logical_deduction_five_objects`, `logical_deduction_seven_objects`, `test`. `qN` is the zero-based row within each config. BBH-3 retains the evaluation's `question`/`answer` schema; BBH-5/7 retain source `input`/`target`.
- **IFEval:** Zhou et al., [Google Research IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval), Apache-2.0. Evaluated [wis-k/instruction-following-eval revision](https://huggingface.co/datasets/wis-k/instruction-following-eval/tree/5a5661c2a35488308556cf4453dc074d1eba91a0) `5a5661c2a35488308556cf4453dc074d1eba91a0`, `train` (used as evaluation split), native `key`. Retain this evaluated version, including its null-valued kwargs, rather than replacing it with a newer upstream input.
- **ScienceQA:** Lu et al., [ScienceQA](https://github.com/lupantech/ScienceQA), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); attribution, noncommercial use and share-alike apply to the released question file. Evaluated [lmms-lab/ScienceQA revision](https://huggingface.co/datasets/lmms-lab/ScienceQA/tree/69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50) `69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50`, `ScienceQA-FULL`, `test`. IDs and image filenames come from the [official problem records](https://github.com/lupantech/ScienceQA/blob/2cbf8318e07b9ece895bb2ae605e71e38d623264/data/scienceqa/problems.json) and [test split](https://github.com/lupantech/ScienceQA/blob/2cbf8318e07b9ece895bb2ae605e71e38d623264/data/scienceqa/pid_splits.json). The released file is the text-only evaluation: image filenames are kept as metadata, but no images are included.
- **HumanEval / MBPP:** Liu et al., [EvalPlus](https://github.com/evalplus/evalplus), Apache-2.0. Exact evaluated cache files `HumanEvalPlus-v0.1.10.jsonl` (MD5 `fe585eb4df8c88d844eeb463ea4d0302`) and `MbppPlus-v0.2.0.jsonl` (MD5 `ee43ecabebf20deef4bb776a405ac5b1`) supply the records, retaining task IDs, prompts, canonical solutions and all existing tests/metadata, with an added release `item_id`. These source-file hashes match saved evaluation results; the released JSONL files have different hashes after adding keys. [Original HumanEval](https://github.com/openai/human-eval) is MIT, copyright OpenAI; [original MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) is CC BY 4.0 (Austin et al., Google Research). EvalPlus's existing changes are preserved; this release only adds the primary key to those inputs.
- **MedQA:** Jin et al., [MedQA](https://github.com/jind11/MedQA), MIT, copyright 2022 Di Jin. Evaluated [lmms-lab/MEDQA revision](https://huggingface.co/datasets/lmms-lab/MEDQA/tree/5ce9484de39cb8085d8febc4cba97c0c6622c6ab) `5ce9484de39cb8085d8febc4cba97c0c6622c6ab`, `default`, `test`, 1,273 five-choice items. `qN` is the zero-based row of that fixed split.
- **ZebraLogic:** Lin et al., [WildEval/ZebraLogic revision](https://huggingface.co/datasets/WildEval/ZebraLogic/tree/0a473f5a0054835754ed156d5a79c6ce27178bb1) `0a473f5a0054835754ed156d5a79c6ce27178bb1`, `grid_mode`, `test`. The 1,000 records are restored from the locally cached Arrow dataset and verified against the cached original Parquet file and saved evaluation inputs. All source fields (`id`, `size`, `puzzle`, `solution`, `created_at`) are retained without rewriting; `difficulty.csv` joins on the added release `item_id`; native `id` is retained. This is the grid-mode dataset, not the multiple-choice variant. A redistribution license was not found in the source dataset card; the requested local copy does not establish a new license.

## Agent benchmarks (task IDs only)

No questions, answers, attachments or traces are included for these benchmarks.

- **SWE-bench-Mini:** all 50 native instance IDs match the [pinned HAL task list](https://github.com/princeton-pli/hal-harness/blob/16bb03ebc11577fb5ea6dc8bb6c968387085e6aa/hal/benchmarks/swebench_verified_mini_task_ids.txt).
- **CORE-Bench Hard:** all 45 native capsule IDs match the [HAL benchmark task identifiers](https://hal.cs.princeton.edu/corebench_hard); the checked public run also matches all 45 local response outcomes by ID.
- **USACO:** all 307 native problem IDs match the [HAL benchmark task identifiers](https://hal.cs.princeton.edu/usaco); the previously checked ten public model runs match the local response outcomes by ID.
- **GAIA — unresolved official mapping:** `source_item_id` strings `"0"` through `"164"` are positions in the archived response matrix, **not official GAIA task UUIDs or verified source-dataset row numbers**. A focused local search did not locate the original response-matrix construction script or a saved index-to-task-ID mapping. These IDs identify only the archived difficulty entries; they cannot currently be used to retrieve official GAIA tasks. No UUIDs have been inferred or substituted.

## License notices

These notices cover the corresponding upstream material described above. They do not establish additional rights beyond the stated source terms. Creative Commons sources retain their respective [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) or [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) terms and attributions above.

<details>
<summary>MIT notices</summary>

Copyright notices: MATH-500/PRM800K — Copyright (c) 2023 OpenAI; HumanEval — Copyright (c) OpenAI (https://openai.com); evaluation input distribution — Copyright (c) 2024 Zhibin Gou; OlympiadBench — Copyright (c) 2024 OpenBMB; BIG-Bench Hard — Copyright (c) 2022 suzgunmirac; MedQA — Copyright (c) 2022 Di Jin. MMLU-Pro attribution: TIGER-Lab and the MMLU-Pro authors (MIT declaration in the pinned dataset card).

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>

<details>
<summary>Apache License 2.0 (EvalPlus, IFEval, and the upstream declarations noted above)</summary>

Attribution: Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang (EvalPlus); Google Research (IFEval).

```text
Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

</details>

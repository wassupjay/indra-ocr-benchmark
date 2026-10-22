# Medium publishing kit

Research checked October 22, 2026. The changes below are editorial choices informed by the linked sources, not measured traffic improvements. No views, ranking or Boost is guaranteed.

## Ready-to-publish settings

**Display title:** Indra OCR Beat Tesseract, EasyOCR & PaddleOCR in Our CPU Speed Test

**Subtitle:** 114 pages, three passes each: Tesseract 5, EasyOCR and PaddleOCR PP-OCRv6 Tiny, tested on one CPU.

**Search-oriented alternative:** Indra vs Tesseract vs EasyOCR vs PaddleOCR: CPU Benchmark

**Suggested description:** Indra, Tesseract, EasyOCR and PaddleOCR Tiny compared on 114 pages: CPU speed, RAM, two reference checks and the limits of a single-book benchmark.

**Relevant topics:** OCR, Computer Vision, Python, Machine Learning, Software Engineering. Choose only matching available topics. These are suggested labels, not tags already applied to a live story.

**Preview image:** chart-speed.png. It must be inserted in the story before Medium can feature it. Select it in the publishing preview; check cropping and focal point on mobile as well as desktop. [Medium's featured-image instructions](https://help.medium.com/hc/en-us/articles/215680047-Setting-a-featured-image-on-your-post).

## What was applied, and why

| Change in the draft | Research basis |
|---|---|
| Headline leads with Indra OCR versus the named alternatives; subtitle states the measured speed/RAM trade-off. | Medium discourages misleading or vague headlines and expects the title, subtitle and image to represent the story. [Distribution guidelines](https://help.medium.com/hc/en-us/articles/360006362473-Medium-s-Distribution-Guidelines-How-curators-review-stories-for-Boost-General-and-Network-Distribution). |
| First chart appears before the hardware/model details. Opening gives the result immediately. | NN/G's reading research supports clear subheadings and easy scanning. This is usability evidence, not a Medium-view-growth experiment. [Layer-cake scanning research](https://www.nngroup.com/articles/layer-cake-pattern-scanning/). |
| Three charts remain, with short takeaway headings, captions and alt text; methods are below the findings. | Applied the same scanning principle without dropping the reference/timing limitations. |
| One relevant closing question asks which document types readers want tested. | Editorial hypothesis: invites useful discussion tied to the experiment. Not a documented ranking factor or proven engagement boost. |

## Transfer to Medium

Use MEDIUM_POST.md or medium-post.html. Upload chart-speed.png, chart-word-errors.png and chart-memory.png in that order. Preserve alt text and captions. Local image paths do not upload automatically. Check the title/subtitle/featured-image preview before publishing; Medium permits custom preview text. [Display-title controls](https://help.medium.com/hc/en-us/articles/214895188-Custom-titles-subtitles).

Replace the local full-methods link with the actual public GitHub file URL once the repository exists. Do not publish a local filename link or invent a repository URL. The detailed version remains in MEDIUM_FULL_METHODS.md; measurements and charts have not changed.

## Distribution beyond the headline

An appropriate technical publication can expose the story to its followers. Read its submission rules first; acceptance and publication timing belong to its editors. [Medium submission instructions](https://help.medium.com/hc/en-us/articles/213904978-How-to-submit-a-story-to-a-publication).

After publication, optionally share one chart and a short finding with an existing relevant audience, linking to the actual story. Avoid unrelated tags, repetitive promotion or sending messages without permission. No submissions or messages have been sent here. No publishing-time trick or keyword density is claimed to guarantee distribution.

**Unsent share copy:** Four OCR engines, the same 114 pages, CPU only. Indra led on throughput and text agreement; Tesseract used the least RAM. Three charts show the trade-offs, including two unverified reference checks and an unexplained timing difference. Add the real Medium URL after publication.

## Measure attention after publishing

Use the story's native detailed Stats page. Record presentations, views, feed click-through rate when available, reads/read ratio, traffic sources and responses/follows. A Medium read means at least 30 seconds, not necessarily finishing the article. [Official stats definitions](https://help.medium.com/hc/en-us/articles/34831991136151-Story-s-detailed-stats-page).

Suggested review cadence: days 1, 7 and 28 (our recommendation, not a Medium rule). Low feed CTR suggests reviewing the preview; views without reads suggest reviewing the opening. These are diagnostic hypotheses. Check traffic sources and audience changes before attributing improvement to an edit. Changing one item at a time is observational, not a controlled A/B test. No live analytics are available before the story is published.

## Authorship and claim limits

Medium's guidance values original human storytelling. AI-generated content can limit wider distribution; disclosing AI assistance does not guarantee eligibility. Review the wording in your own voice using only actual experience. Keep the AI disclosure within the first two paragraphs and the image captions. A substantially AI-generated draft must not be paywalled under the Partner Program. [Medium AI policy](https://help.medium.com/hc/en-us/articles/22576852947223-Artificial-Intelligence-AI-content-policy).

Keep both references labeled unverified, retain the unexplained Indra timing drift, and do not call the i7/96 GiB system entry-level. This experiment does not establish a universal fastest or most accurate OCR engine. Nothing has been published or changed on Medium by this edit.


## Independent writers: additional research and applied choices

These are outside Medium's official help pages. Their results are self-reported and mostly observational. Older articles describe past versions of Medium; even the newer audit does not establish how today's algorithm works. No reported traffic increase is a forecast for this OCR article.

| Independent source | Useful lesson and what we did |
|---|---|
| [Quincy Larson, freeCodeCamp (2016)](https://www.freecodecamp.org/news/how-to-write-medium-stories-people-will-actually-read-92e58a27c8d8/) | Prioritize the headline/image combination, short paragraphs and an audience-relevant story. Applied a concrete test-count headline and retained our own data chart as the preview image. We did not copy emotional/profane headline formulas or treat an old word-count recommendation as a ranking rule. |
| [Mark Thompson, 268-story audit (May 2026)](https://thesystematicwriter.substack.com/p/how-medium-works-possibly-a-268-story) | The author reports that first-person/specific-number titles worked in his own collection, while a few outliers dominated results. The test size and chart count remain in the article; the author-selected main title leads with the engine comparison. No invented personal experience or causal algorithm claim. |
| [Clay Ostrom, Map and Fire (updated 2020)](https://mapandfire.com/blog/what-i-learned-from-publishing-the-same-article-twice/) | A title/image republishing experiment attracted more attention but weaker read-through; publication placement also changed. Keep the preview aligned with the article's promise and inspect reads as well as views. We are not duplicating the post or claiming a controlled headline test. |
| [Maarten Dalmijn, niche-audience account (2021)](https://mdalmijn.com/p/how-i-reached-32k-followers-with-a-niche-audience) | Develop a useful perspective for a particular readership and learn from feedback over time. The opening now centers a practical developer decision: CPU wait time versus RAM. The reference-change finding supplies a specific lesson beyond a generic engine ranking. We did not turn the book title into a misleading personal-finance hook. |
| [Niki Tisza, 100-comment experiment (2023)](https://www.nikitisza.com/writing/in-7-days-i-left-100-comments-on-medium-stories-and-this-is-what-happened-to-my-views) | The writer reports engagement gains but also substantial time cost and measurement limitations. This does not justify automated comments, engagement exchanges or a 100-comment quota. No comments or other outreach have been posted. |

**Current headline hypothesis:** the explicit Indra OCR comparison may help readers recognize the topic quickly. The author requested this engine-first title; it is not a measured traffic winner. This has not been A/B-tested. The prior results-first title remains an alternative: ?CPU OCR Benchmark: Indra Was Fastest, Tesseract Used the Least RAM.? If testing after publication, change one element at a time, record dates and traffic sources, and treat differences as observational.

**Applied in the body:** a reader-problem opening, first chart before setup, short takeaway headings, a reference-change insight, one relevant closing question, and preserved limitations. The three charts and all measurements are unchanged. No unsupported best-posting-hour, keyword-density, paid-clap or guaranteed-viral tactics were applied.

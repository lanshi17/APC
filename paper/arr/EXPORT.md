# ARR / arXiv export notes

## ARR (Oct 12, 2026)
- Submit: main.pdf (review mode: anonymized, line numbers) + refs.bib + main.tex source
- Responsible NLP checklist answers: arr_checklist_answers.md (copy into OpenReview form)
- Title/authors: main.tex is anonymous; OpenReview author field = real names (not in PDF)

## arXiv
- main-arxiv.pdf ([final] mode, no line numbers) — FILL the author placeholder in
  main-arxiv.tex first; consider adding acknowledgment of the gateway vendor only if
  authorship permits (de-anonymization).
- Suggested primary category: cs.CL; cross-list cs.LG
- Version note: v1 = this manuscript; update on any number-affecting rerun.

## Camera-ready TODO if accepted (NAACL/COLING 2027)
- switch to venue .sty if required (acl.sty works for both NAACL/COLING via ACL family)
- renumber sections only if the venue template forbids \S self-references (current: fine)
- regenerate tables from experiments/*.json if any number changes after camera audit

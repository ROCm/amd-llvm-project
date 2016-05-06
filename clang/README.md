HCC Clang frontend upgrade repository
=====================================

This repository holds the work of upgrading clang frontend used in HCC from
3.5 to tip of tree. The goal is to align the version used in the frontend and 
backend.

Branches
========
- master : holds production-ready codes

- upstream : holds commits from upstream clang
  The branch always have the latest vanilla clang.

- develop : holds hcc-specific codes
  Developments are always conducted here.

- release_YYWW : release branches for week WW year YY
  Periodically release branches would be created to merge all latest commits
  upstream and develop branch. Once tested, it would be promoted to master.


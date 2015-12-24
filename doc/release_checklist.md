This instruction is after the coding phase. First make sure that the code for the new release pass every test and is frozen.

- update release notes (and related docs if necessary).
- update version info in CMakeLists.txt:

  `set(gunrock_VERSION_MAJOR major_num)`
  
  `set(gunrock_VERSION_MINOR minor_num)`
  
  `set(gunrock_VERSION_PATCH patch_num)`
- update doxygen version info in doc/gunrock.doxygen
  
  `PROJECT_NUMBER = version_number`
- run 'doxygen gunrock.doxygen' in gunrock_dir/doc to generate new docs (remember to remove **ALL**
  the warnings in the warn.log file)
  
- Update gh-pages branch

  `git checkout gh-pages`
  
  rename generated html: `cd doc && mv html version_number` (for example, for 0.4 release, `mv html 0.4`

  edit the index.html if necessary (TODO: find a smart way to sync README.md and gh-page's index.html automatically.)
  
  `git add -f ./version_number/* && git commit -am "updated docs" && git push origin gh-pages`
  
- Sync dev with master
  `git checkout master`
  
  `git pull origin dev`
  
  merge conflict
  
  `git commit -am "merged conflict"`
  
  `git pull`
  
  `git push origin master`
  
- Create the release tag and new release.

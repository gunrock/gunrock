Release checklist without the automation script
==============

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
  
  rename generated html: `cd doc && mv html version_number` (for example, for 0.4 release, `mv html 0.4`)

  make a symbolic link from `latest` to the version number you are releasing (for example, `ln -sf 0.4 latest`)

  If you have additional resources (like Vega graphs), remember to copy them to gh-pages branch as well. 
  The location will depend on how you referenced those resources in the generated HTML.

  ~~edit the index.html if necessary (TODO: find a smart way to sync README.md and gh-page's index.html automatically.)~~
  
  `git add -f ./version_number/* && git commit -am "updated docs" && git push origin gh-pages`
  
- Sync dev with master
  `git checkout master`
  
  `git pull origin dev`
  
  merge conflict
  
  `git commit -am "merged conflict"`
  
  `git pull`
  
  `git push origin master`
  
- Create the release tag and new release.

New release with the documentation automation script
==============
A bash script releasedoc.sh is provided to automate most of the steps above.

Basically, you still need to update CMakeLists.txt manually (this serves as the
global version number). Run ./releasedoc.sh in the 'docs' folder, it will
automatically update PROJECT_NUMBER in gunrock.doxygen, 
run doxygen, check if there are warnings, create a new folder for the latest 
version of documentation inside the gh-pages branch with freshly generated
documentation pages and point the "latest" symbolic link to it. 
It will commit those changes, but won't push the new commit to the 
upstream; you still need to manually do that after verifying everything is good.

The following warnings during the release process can be safely ignored:

```
warning: unable to rmdir externals/cub: Directory not empty
warning: unable to rmdir externals/moderngpu: Directory not empty
```

It is the expected behavior of git with external repositories (in this case,
cub and moderngpu).

Adding a new page
==============

To add a new page, just create a new Markdown file and write the things you want,
then add that file to the "INPUT" list in gunrock.doxygen (near line 680).
To preview the new page, run `doxygen gunrock.doxygen` and point your browser to
docs/html/pages.html. When you are ready, run ./releasedoc.sh and that page will
be added to documentation of the latest version of Gunrock.

If you want to add Vega graphs in your Markdown, please take a look at the examples in
vegademos.md. A vega graph is just a JavaScript snippet inside any Markdown files 
that are used by Doxygen. The easiest example would be: 

```
\htmlonly
<div id="verticalbar"></div>
<script type="text/javascript">
plotjson("verticalbar", "../graphs/vertical_bar.json");
</script>
\endhtmlonly
```

The JSON files should be put into the "docs/graphs" folder, and they should be 
referenced as "../graphs/file_name.json". 
releasedoc.sh will copy them to the "graphs" folder inside gh-pages for you.

For more examples on using a Vega graph, please take a look at `doc/vegademo.md`.

Making changes to the documentation pages of an existing release
==============
Just modify the Markdowns as you wish. To preview, run `doxygen gunrock.doxygen` 
and point your browser to the page you modified. When you are ready, 
just run ./releasedoc.sh and that page will
be updated in the documentation of the latest version of Gunrock.

In this case, because a folder containing documentation for the current version 
exists in gh-pages branch, releasedoc.sh will remove it and replace it with all 
freshly generated pages with new changes. Although we completely remove the old 
documentation folder and replace it with a new one, git will be smart enough to 
figure out the differences between old and new HTML files, and only make an 
incremental update. Please note that the timestamps on all generated HTML pages
will change after you run releasedoc.sh.

If you just need to update a Vega graph, you can just update the JSON inside
"graphs" folder, and then run ./releasedoc.sh (or manually copy your new JSON
to the "graphs" folder in the gh-pages branch and overwrite the old one).


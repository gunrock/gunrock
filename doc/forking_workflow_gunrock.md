Git Forking Workflow
================

Transitioning over from [Git Branching Workflow](http://nvie.com/posts/a-successful-git-branching-model/) suggested by Vincent Driessen at nvie to [Git Forking Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows#forking-workflow) for Gunrock.

How Forking Workflow Works?
=============
![Forking Workflow](https://wac-cdn.atlassian.com/dam/jcr:5c0941ff-a8b5-435b-a092-2167705f1e97/01.svg?cdnVersion=hp)
> As in the other Git workflows, the Forking Workflow begins with an **official public repository** stored on a server. But when a new developer wants to start working on the project, they do not directly clone the official repository.
<br><br>Instead, they **fork the official repository** to create a copy of it on the server. This new copy serves as their personal public repository—no other developers are allowed to push to it, but they can **pull changes** from it (we’ll see why this is important in a moment). After they have created their server-side copy, the developer performs a git clone to get a copy of it onto their local machine. This serves as their private development environment, just like in the other workflows.
<br><br> When they're ready to publish a local commit, they push the commit to their own public repository—not the official one. Then, they file a pull request with the main repository, which lets the project maintainer know that an update is ready to be integrated. The **pull request also serves as a convenient discussion thread** if there are issues with the contributed code.
<br><br> To integrate the feature into the official codebase, the maintainer pulls the contributor’s changes into their local repository, checks to make sure it doesn’t break the project, merges it into his local master branch, then pushes the master branch to the official repository on the server. The contribution is now part of the project, and other developers should pull from the official repository to synchronize their local repositories.

Gunrock's Forking Workflow:
=============

**gunrock/gunrock:**
* **Master Branch:** Reserved only for final releases or some bug fixes/patched codes.
* **Dev Branch:** Current working branch where all developers push their changes to. This dev branch will serve as the "next release" gunrock, eliminating the need of managing individual branches for each feature and merging them when it is time for the release.


**personal-fork/gunrock**
* **Feature Branch:** This is the developer's personal repository with their feature branch. Whatever changes they would like to contribute to gunrock must be in their own personal fork. And once it is time to create a pull request, it is done so using github pull request, a reviewer checks it and the changes are merged into gunrock/gunrock dev branch.


Note that transitioning to this type of workflow from branching model doesn't require much effort, we will just have to start working on our forks and start creating pull requests to one dev branch.

How to contribute?
=============
* Fork using GitHub; https://github.com/gunrock/gunrock
* `git clone --recursive https://github.com/gunrock/gunrock.git`
* `git remote set-url --push origin https://github.com/username/gunrock.git` This insures that you are pulling from `gunrock/gunrock` (staying updated with the main repository) but pushing to your own fork `username/gunrock`.
* `git add <filename>`
* `git commit -m "Describe your changes."`
* `git push`
* Once you've pushed the changes on your fork, you can create a **pull request** on Github to merge the changes.
* Pull request will then be reviewed and merged into the `dev` branch.

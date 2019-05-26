1. understanding Revert
  1.1 Temporarily switch to a different commit
    If you want to temporarily go back to it, fool around, then come back to where you are,
    all you have to do is check out the desired commit:

# This will detach your HEAD, that is, leave you with no branch checked out:
    git checkout 0d1d7fc32
    git checkout -b old-state 0d1d7fc32

  1.2 (If you've made changes, as always when switching branches, you'll have to deal with
    them as appropriate. You could reset to throw them away; you could stash, checkout,
    stash pop to take them with you; you could commit them to a branch there if you want
    a branch there.)

  1.3 Hard delete unpublished commits
    If, on the other hand, you want to really get rid of everything you've done since
    then, there are two possibilities. One, if you haven't published any of these commits,
    simply reset:

    # This will destroy any local modifications.
    # Don't do it if you have uncommitted work you want to keep.
    git reset --hard 0d1d7fc32

    # Alternatively, if there's work to keep:
    git stash
    git reset --hard 0d1d7fc32
    git stash pop
    # This saves the modifications, then reapplies that patch after resetting.
    # You could get merge conflicts, if you've modified things which were
    # changed since the commit you reset to.

  1.4  If you mess up, you've already thrown away your local changes, but you can at least get back to where you were before by resetting again.

  Undo published commits with new commits
  On the other hand, if you've published the work, you probably don't want to reset the branch, since that's effectively rewriting history.
  In that case, you could indeed revert the commits. With Git, revert has a very specific meaning: create a commit with the reverse patch to
  cancel it out. This way you don't rewrite any history.

  # This will create three separate revert commits:
  git revert a867b4af 25eee4ca 0766c053

  # It also takes ranges. This will revert the last two commits:
  git revert HEAD~2..HEAD

  #Similarly, you can revert a range of commits using commit hashes:
  git revert a867b4af..0766c053

  # Reverting a merge commit
  git revert -m 1 <merge_commit_sha>

  # To get just one, you could use `rebase -i` to squash them afterwards
  # Or, you could do it manually (be sure to do this at top level of the repo)
  # get your index and work tree into the desired state, without changing HEAD:
  git checkout 0d1d7fc32 .

  # Then commit. Be sure and write a good message describing what you just did
  git commit
  The git-revert manpage actually covers a lot of this in its description. Another useful link is this git-scm.com section discussing git-revert.

If you decide you didn't want to revert after all, you can revert the revert (as described here) or reset back to before the revert (see the previous section).

Remote:

A-----C----E ("stable")
 \   (master)
  B-----D-----F----- ("new-idea")
    local uncomitted changes

2. git branch -r (remote)
   git checkout [-b] branch-name
   git merge xxx(merge xxx to your current local branch)
   git fetch, git merge()

   git push origin master --force(pushing your local master to remote origin)
   git push origin experimental:experiment-by-bob
3. git filter-branch --tree-filter 'rm path/to/your/bigfile' HEAD

4. Correct ways to commit and push
  git diff master origin/master
  git fetch
  git merge origin/master(merge from remote)
  git add --all .
  git commit -m

5. remove files
  git rm --cached $FILE
  echo $FILE >> .gitignore
  git add .gitignore
  git commit --amend --no-edit
  git reflog expire --expire=now --all && git gc --prune=now --aggressive

6. check local commits
git log --branches --not --remotes
git-show-branch - Show branches and their commits

7. remember your username and password
git config credential.helper store

If I have committed my local files, but didn't publish them into remote,
and then I tried to commit new changes, but delete some of the original files,
then if I do push, will it cancel the previous commits? I have to revert my previous
commit, and add new correct files into the commit streams.

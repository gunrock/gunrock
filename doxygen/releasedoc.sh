#!/bin/bash

##############################################################################
##
##  Documentation automation script for Gunrock
##
##  Please run this script in 'doc' folder, inside master/dev branch
##  This script requires GNU Coreutils to run correctly
##
##  Maintainer: Huan Zhang <ecezhang@ucdavis.edu>
##
##############################################################################

#----------------------------------------------------------------------------#
# Definitions of paths
#----------------------------------------------------------------------------#
CMAKELISTS="../CMakeLists.txt"
DOXYGENEXE="doxygen"
DOXYGENFILE="gunrock.doxygen"
LOGFILE="doxygen.log"
# JSONs for Vega plots should be put into the "graphs" directory
GRAPHSDIR="graphs"

#----------------------------------------------------------------------------#
# Colored terminal ouput
#----------------------------------------------------------------------------#
COLOR_K='\033[1;30m'
COLOR_R='\033[1;31m'
COLOR_G='\033[1;32m'
COLOR_Y='\033[1;33m'
COLOR_B='\033[1m'
COLOR_M='\033[1;35m'
COLOR_C='\033[1;36m'
COLOR_OFF='\033[m'

highlighted () {
        local message=$1
        echo -e -n $COLOR_B
        echo -e "$message"
        echo -e -n $COLOR_OFF
}

warn () {
        local message=$1
        echo -e -n $COLOR_M
        echo -e "$message"
        echo -e -n $COLOR_OFF
}

error () {
        local message=$1
        echo -e -n $COLOR_R
        echo -e "$message"
        echo -e -n $COLOR_OFF
        exit
}

#----------------------------------------------------------------------------#
# Utility functions for extracting verion information from CMakeLists.txt
#----------------------------------------------------------------------------#
extract_version () {
        local cmakelists=$1
        local version_major=""; local version_minor=""
        version_major=$($GREP -Po 'gunrock_VERSION_MAJOR \K[0-9]+' $cmakelists) || return 1
        version_minor=$($GREP -Po 'gunrock_VERSION_MINOR \K[0-9]+' $cmakelists) || return 1
        echo $version_major.$version_minor
        return 0
}

#----------------------------------------------------------------------------#
# Utility functions for obtaining/updating doxygen configuration
#----------------------------------------------------------------------------#
update_doxygen_variable () {
        local doxygenfile=$1
        local doxygen_variable=$2
        local value=$3
        $SED -i'' 's/\('"$doxygen_variable"'[[:space:]]*=[[:space:]]*\).*/\1'$value'/g' $doxygenfile
}

get_doxygen_variable () {
        local doxygenfile=$1
        local doxygen_variable=$2
        $GREP -Po "$doxygen_variable"'[[:space:]]*=[[:space:]]*\K.*' $doxygenfile || return 1
        return 0
}

#----------------------------------------------------------------------------#
# Utility functions for comparing version numbers
#----------------------------------------------------------------------------#

version_compare_lte () {
        [ "$1" == "$(echo -e "$1\n$2" | $SORT -V | head -n1)" ]
}

version_compare_lt () {
        [ "$1" == "$2" ] && return 1 || version_compare_lte $1 $2
}

#----------------------------------------------------------------------------#
# Utility functions for checking required tools (especially for OS X)
#----------------------------------------------------------------------------#

check_tools () {
        local var_name=$(echo $1 | tr '[:lower:]' '[:upper:]')
        local tool_name=$(echo $1 | tr '[:upper:]' '[:lower:]')
        echo -n "Checking for $tool_name..."
        eval "$var_name=\"\""
        hash "$tool_name" 2> /dev/null && eval "$var_name=$tool_name"
        hash "g$tool_name" 2> /dev/null && eval "$var_name=g$tool_name"
        highlighted "${!var_name}"
        [ -z "${!var_name}" ] && error "Please install GNU $tool_name"
}

#----------------------------------------------------------------------------#
# Main script
#----------------------------------------------------------------------------#

# OS and tools check

REQUIRED_TOOLS=("grep" "sort" "sed" "find")

echo -n "We are running on "
os=$(uname -s)
highlighted $os
for i in ${!REQUIRED_TOOLS[@]}; do
        check_tools "${REQUIRED_TOOLS[i]}"
done

# Check if we are inside master or dev branch

echo -n "We are running in branch "
current_branch=$(git rev-parse --abbrev-ref HEAD)
highlighted $current_branch
[ "$current_branch" != "master" -a "$current_branch" != "pre-release" ] && error "Please run this script inside master or dev branch!"


# Extract the latest version number from CMakeLists.txt

echo -n "Checking gunrock version number..."
gunrock_version=$(extract_version "$CMAKELISTS") || error "Can't find version numbers in $CMAKELISTS. Please run this script in 'doc' folder!"
highlighted "$gunrock_version"

# Update PROJECT_NUMBER in gunrock.doxygen based on the number in CMakeLists.txt

echo -n "Replacing doxygen project version number..."
old_version=$(get_doxygen_variable "$DOXYGENFILE" "PROJECT_NUMBER") || error "Can't find PROJECT_NUMBER in $DOXYGENFILE"
[ "$old_version" != "$gunrock_version" ] && update_doxygen_variable "$DOXYGENFILE" "PROJECT_NUMBER" "$gunrock_version"
[ "$old_version" == "$gunrock_version" ] && highlighted "version unchanged" || highlighted "updated!"

# Run doxygen...

doxygen_output_path=$(get_doxygen_variable $DOXYGENFILE "HTML_OUTPUT")
echo -n "Generating Doxygen pages in folder \"$doxygen_output_path\"..."
"$DOXYGENEXE" "$DOXYGENFILE" > $LOGFILE 2>&1 || error "Doxygen failed. Please check $LOGFILE"

# Warn you if there are doxygen warnings

warn_file=$(get_doxygen_variable $DOXYGENFILE "WARN_LOGFILE")
num_warnings=$(cat "$warn_file" | $GREP "warning" | wc -l)
[ "$num_warnings" == "0" ] && highlighted "No warnings ;)" || warn "$num_warnings warnings...see $warn_file"
rm $LOGFILE

# Copying Vega graph JSON folder into the generated folder

echo -n "Copying Vega graphs..."
if [ "$(ls -A $GRAPHSDIR 2>/dev/null)" ]
then
        cp -r "$GRAPHSDIR" "$doxygen_output_path"/
        highlighted "done"
        graphs_exist=true
else
        highlighted "skipped ($GRAPHSDIR folder is empty)"
        graphs_exist=false
fi

# Switch to gh-pages branch, and check if documentation for that version exists or not

echo "Switching to gh-pages (warnings on external submodules can be ignored)..."
git stash > /dev/null 2>&1;
git checkout gh-pages > /dev/null || error "Cannot switch to gh-pages branch!"
prev_doc_revs=$($FIND ./ -maxdepth 1 -type d -printf "%f\n" | $GREP "[0-9][0-9]*\.[0-9][0-9]*" | tr '\n' ' ')
prev_doc_latest=$($FIND ./ -maxdepth 1 -type d -printf "%f\n" | $GREP "[0-9][0-9]*\.[0-9][0-9]*" | $SORT | tail -n 1)
echo -n "Found old documentation for $prev_doc_revs- "
[[ $prev_doc_revs == *"$gunrock_version"* ]] && highlighted "I will replace $gunrock_version!" || highlighted "I will create a new folder for $gunrock_version"

# If documentation for the same version exists, remove it (after user's confirmation)

[[ $prev_doc_revs == *"$gunrock_version"* ]] && read -p "I will remove the folder containing old documentation for $gunrock_version. Proceed? [Y/n] " -r users_anwser
users_anwser=${users_anwser:-"Y"}
if ! [[ $users_anwser =~ ^[Yy]$ ]]
then
        git checkout "$current_branch" > /dev/null
        git stash apply > /dev/null 2>&1
        error "Exiting as requested by user's choice \"$users_anwser\""
fi

# Update the 'latest' symbolink if we are generating docs for lastest version

rm -rf "$gunrock_version"
mv "$doxygen_output_path" "$gunrock_version"
$graphs_exist && rsync -ac "$gunrock_version"/"$GRAPHSDIR"/ "$GRAPHSDIR"
if version_compare_lt $gunrock_version $prev_doc_latest
then
        echo "We are not updating the latest docs. No need to update symbolink."
else
        echo "We are updating the latest docs. Will update symbolink."
        rm "latest"
        ln -sv "$gunrock_version" "latest"
fi
echo -n "Latest documentation have been updated to folder "; highlighted $gunrock_version

# Add things to git and commit them

echo "Now run git-commit..."
git add -f "$gunrock_version"
git add -f latest
$graphs_exist && git add -f "$GRAPHSDIR"
git commit -e -am "updated docs for $gunrock_version"

# If this commit failed (for example, due to empty commit message), give user the option to reset

if [ $? -ne 0 ]
then
        read -p "Commit aborted. Reset gh-pages to last commit? [Y/n] " -r users_anwser
        users_anwser=${users_anwser:-"Y"}
        if [[ $users_anwser =~ ^[Yy]$ ]]
        then
                git reset HEAD --hard
        fi
fi

# switch back to original branch

git checkout "$current_branch" > /dev/null || error "Cannot switch back to $current_branch"
git stash apply > /dev/null 2>&1
git stash drop > /dev/null 2>&1
highlighted "Done! Please push after you verified everything looks good."

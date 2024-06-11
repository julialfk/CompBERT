module Main

import IO;
import lang::java::m3::Core;
import lang::java::m3::AST;
import lang::json::IO;
import util::FileSystem;
import util::Math;
import List;
import Map;
import Type;
import Extract;

// Run the analysis over multiple Java programs.
// int mainList(list[loc] projectlocations) {
//     for (project <- projectlocations) {
//         main(project);
//     }

//     return 0;
// }

// 
int main(loc projectLocation, loc issuesLocation) {
    map[str, map[str, value]] issues = readJSON(#map[str, map[str, value]], issuesLocation);

    set[str] commits = {};
    set[str] dupeCommits = {};
    list[map[str, value]] entries = [];
    for (issue <- issues) {
        iprintln(issue);

        // If any of the patches is also found in a parent commit,
        // it should be excluded as an old_code entry,
        // since it cannot be used as a negative training example
        list[node] nodesNew = [];
        list[map[str, value]] methodsOldAll = [];
        map[str, map[str, map[str, list[map[str, value]]]]] methods = ();
        for (commit <- typeCast(#list[str], issues[issue]["commits"])) {
            if (!exists(projectLocation + commit)) { continue; }

            if (commit in commits) { dupeCommits += commit; }
            else { commits += commit; }

            list[map[str, value]] methodsNew = [];
            list[map[str, value]] methodsOld = [];
            map[str, value] baseInfo = ("issue":issue,
                                        "commit":commit,
                                        "summary":issues[issue]["summary"],
                                        "description":issues[issue]["description"]);
            <nodesNew, methodsNew, methodsOld> =
                getChangedMethods(projectLocation + commit,
                                    nodesNew,
                                    methodsOld,
                                    baseInfo);
            entries += methodsNew;
            methodsOldAll += methodsOld;
        }
        issues[issue]["positives"] = size(nodesNew);

        // Only add an unchanged method as an entry if they have not been marked as
        // changed in a previous commit for the same issue
        int nNegatives = 0;
        iprintln("#Methods to double check:" + toString(size(methodsOldAll)));
        for (method <- methodsOldAll) {
            if (!any(n <- nodesNew, n:= method["node"])) {
                entries += delete(method, "node");
                nNegatives += 1;
            } // Add check to ensure negs only appear once in set by also matching
            // to set of nodes that grows when neg is added (node of neg is added after neg is written)
        }
        issues[issue]["negatives"] = nNegatives;
    }

    writeJSON(projectLocation + "data.json", entries);
    writeJSON(issuesLocation, issues);
    writeJSON(projectLocation + "duplicate_commits.json", dupeCommits);

    return 0;
}
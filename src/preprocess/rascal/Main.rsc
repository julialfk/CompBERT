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
import Compare;


map[str, list[node]] updateNodes(map[str, value] method, map[str, list[node]] methodNodes) {
    methodName = typeCast(#str, method["method_name"]);
    methodNode = method["node"];

    if (methodName in methodNodes) {
        if (any(n <- methodNodes[methodName], areSimilar(n, methodNode))) { return (); }
        methodNodes[methodName] += methodNode;
    }
    else { methodNodes += (methodName:[methodNode]); }

    return methodNodes;
}


/**
 * Main function to analyze commits and methods for issues in a project.
 *
 * @param projectLocation The location of the project directory.
 * @param issuesLocation The location of the issues JSON file.
 * @return An integer status code (0 for success).
 */
int main(loc projectLocation, loc issuesLocation) {
    // Read the issues data from the JSON file
    map[str, map[str, value]] issues = readJSON(#map[str, map[str, value]], issuesLocation);

    loc dataLocation = projectLocation + "data.json";
    writeJSON(dataLocation, []);
    loc methodsOldLoc = projectLocation + "methodsOld_tmp.json";
    loc methodsNewLoc = projectLocation + "methodsNew_tmp.json";

    set[str] commits = {};
    set[str] dupeCommits = {};

    int i = 0;
    println("Total #issues: <size(issues)>");
    for (str issue <- issues) {
        list[map[str, value]] entries = [];
        writeJSON(methodsOldLoc, []);
        writeJSON(methodsNewLoc, []);
        if (i % 10 == 0) { println("i = <i>"); }
        iprintln(issue);

        // Variables to track nodes and methods
        map[str, map[str, map[str, list[map[str, value]]]]] methods = ();
        for (commit <- typeCast(#list[str], issues[issue]["commits"])) {
            if (!exists(projectLocation + commit)) { continue; }

            if (commit in commits) { dupeCommits += commit; }
            else { commits += commit; }

            list[map[str, value]] methodsNew = [];
            list[map[str, value]] methodsOld = [];
            // Base information to be added to each method's info
            map[str, value] baseInfo = ("issue":issue,
                                        "commit":commit,
                                        "summary":issues[issue]["summary"],
                                        "description":issues[issue]["description"],
                                        "nl_input":issues[issue]["nl_input"]);
            // Get changed methods for the commit
            <methodsNew, methodsOld> =
                getChangedMethods(projectLocation + commit, baseInfo);

            list[map[str, value]] methodsOldAll = readJSON(#list[map[str, value]], methodsOldLoc);
            list[map[str, value]] methodsNewAll = readJSON(#list[map[str, value]], methodsNewLoc);
            writeJSON(methodsOldLoc, methodsOldAll + methodsOld);
            writeJSON(methodsNewLoc, methodsNewAll + methodsNew);
        }

        list[map[str, value]] methodsOldAll = readJSON(#list[map[str, value]], methodsOldLoc);
        list[map[str, value]] methodsNewAll = readJSON(#list[map[str, value]], methodsNewLoc);

        // Prevent duplicate new method entries
        map[str, list[node]] nodesNew = ();
        for (method <- methodsNewAll) {
            update = updateNodes(method, nodesNew);
            if (update != ()) {
                nodesNew = update;
                entries += delete(method, "node");
            }
        }
        issues[issue]["positives"] = size(entries);

        // Only add an unchanged method as an entry if they have not been marked as
        // changed in a previous commit for the same issue
        map[str, list[node]] nodesOld = ();
        int nNegatives = 0;
        iprintln("#Methods to double check: " + toString(size(methodsOldAll)));
        for (method <- methodsOldAll) {
            update = updateNodes(method, nodesOld);
            if (update == ()) { continue; }
            else { nodesOld = update; }

            methodName = typeCast(#str, method["method_name"]);
            oldNode = typeCast(#node, method["node"]);
            if (methodName notin nodesNew || !any(n <- nodesNew[methodName], areSimilar(n, oldNode))) {
                entries += delete(method, "node");
            }
        }
        issues[issue]["negatives"] = size(entries) - typeCast(#int, issues[issue]["positives"]);

        i += 1;
        entriesOld = readJSON(#list[map[str, value]], dataLocation);
        writeJSON(dataLocation, entriesOld + entries);
    }

    // Write the entries and issues data to JSON files
    writeJSON(issuesLocation, issues);
    writeJSON(projectLocation + "duplicate_commits.json", dupeCommits);

    return 0;
}
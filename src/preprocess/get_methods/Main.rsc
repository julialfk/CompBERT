module Main

import IO;
import lang::java::m3::Core;
import lang::java::m3::AST;
import lang::json::IO;
import util::FileSystem;
import List;
import Type;
import Extract;

// Run the analysis over multiple Java programs.
// int mainList(list[loc] projectlocations) {
//     for (project <- projectlocations) {
//         main(project);
//     }

//     return 0;
// }

// Create a list of all file locations in the project.
// set[loc] getFiles(loc projectLocation) {
//     // M3 model = createM3FromDirectory(projectLocation);
//     set[loc] fileLocations = files(projectLocation);
//     return fileLocations;
// }

// 
int main(loc projectLocation, loc issuesLocation) {
    map[str, map[str, value]] issues = readJSON(#map[str, map[str, value]], issuesLocation);

    for (issue <- issues) {
        iprintln(issue);
        map[str, map[str, map[str, list[map[str, value]]]]] methods = ();
        for (commit <- typeCast(#list[str], issues[issue]["commits"])) {
            methods[commit] = getChangedMethods(projectLocation + commit);
        }
        issues[issue]["commits"] = methods;
    }

    writeJSON(issuesLocation, issues);

    return 0;
}
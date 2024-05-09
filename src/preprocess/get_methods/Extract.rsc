module Extract

import IO;
import util::FileSystem;
import lang::java::m3::Core;
import lang::java::m3::AST;
import lang::json::IO;
import List;
import String;
import Map;
import Type;

// Create a list of all method locations in the file.
public map[str, map[str, list[map[str, value]]]] getMethods(loc fileLocation,
                                                            map[str, map[str, list[map[str, value]]]] methods,
                                                            map[str, map[str, list[map[str, value]]]] methodsOld,
                                                            map[str, map[str, list[tuple[int,int]]]] diffLines) {
    str fileLocStr = fileLocation.path;
    // Remove the commit specific folders from path
    fileLocStr = visit (fileLocStr) { case /^\/[a-z0-9]+\/(new|old[0-9]+)\// => "" }
    if (fileLocStr notin methods) { methods += (fileLocStr:()); }
    // map[str, map[str, list[map[str, value]]]] methods = (fileLocStr:());

    list[str] fileLines = readFileLines(fileLocation);

    Declaration ast = createAstFromFile(fileLocation, true);
    visit(ast) {
        case a:\method(_,name,_,_,_): {
            codeStr = flattenCode(fileLines, a.src);
            map[str, value] methodInfo = ("start_line":a.src.begin.line,
                                            "end_line":a.src.end.line,
                                            "code":codeStr);

            // If another map of methods from previous commits is passed,
            // compare the new method with the old commits.
            // If they are different, the method has been changed.
            if (!isEmpty(methodsOld)) {
                methodInfo +=
                    ("changed":methodChanged(name, methodInfo, methodsOld, diffLines, fileLocStr));
            }

            if (name in methods[fileLocStr]) { methods[fileLocStr][name] += methodInfo; }
            else { methods[fileLocStr] += (name:[methodInfo]); }
        }
    }

    return methods;
}

// Convert the method's code into a single string.
str flattenCode(list[str] fileLines, loc methodLocation) {
    list[str] methodOriginal =
        fileLines[methodLocation.begin.line-1..methodLocation.end.line];

    str method = "";
    // No need to skip first newline, as the tokenizer will get rid of it.
    for (line <- methodOriginal) {
        method = method + "\n" + line;
    }

    return method;
}

// Compare the current method with previous versions to check if changes have been made
bool methodChanged(str methodName,
                   map[str, value] methodInfo,
                   map[str, map[str, list[map[str, value]]]] methodsOld,
                   map[str, map[str, list[tuple[int,int]]]] diffLines,
                   str fileLocStr) {
    list[tuple[int,int]] diffAfter = diffLines[fileLocStr]["diff_after"];
    // Check if method has lines in any of the shown lines window of the patch
    for (diff <- diffAfter) {
        if (typeCast(#int, methodInfo["start_line"]) <= diff[1]
            && typeCast(#int, methodInfo["end_line"]) >= diff[0]) {
            // Compare the new method with all previous versions
            for (oldInfo <- methodsOld[fileLocStr][methodName]) {
                if (oldInfo["code"] != methodInfo["code"]) { return true; }
            }
        }
    }
    return false;
}

// Create a map containing all changed files from a commit and find which functions
// have been changed
map[str, map[str, list[map[str, value]]]] getChangedMethods(loc commitLocation) {
    set[loc] fileLocationsNew = files(commitLocation + "new");
    set[loc] fileLocationsOld = files(commitLocation) - fileLocationsNew;

    // Read the patch lines from the commit's json
    loc diffLinesLoc = commitLocation + "file_data.json";
    map[str, map[str, list[tuple[int,int]]]] diffLines =
        readJSON(#map[str, map[str, list[tuple[int,int]]]], diffLinesLoc);

    map[str, map[str, list[map[str, value]]]] methodsOld = ();
    for (file <- fileLocationsOld) { methodsOld = getMethods(file, methodsOld, (), ()); }

    map[str, map[str, list[map[str, value]]]] methodsNew = ();
    for (file <- fileLocationsNew) { methodsNew = getMethods(file, methodsNew, methodsOld, diffLines); }

    writeJSON(|project://tmp/changes.json|, methodsNew);
    writeJSON(|project://tmp/old.json|, methodsOld);
    return methodsNew;
}

module Extract

import IO;
import util::FileSystem;
import lang::java::m3::Core;
import lang::java::m3::AST;
import lang::json::IO;
import Node;
import List;
import Map;
import Set;
import String;
import Type;

// Create a list of all method locations in the file.
public map[str, map[str, list[map[str, value]]]]
getMethodsOld(loc fileLocation, map[str, map[str, list[map[str, value]]]] methods) {
    str fileLocStr = fileLocation.path;
    // Remove the commit specific folders from path
    fileLocStr = visit (fileLocStr) { case /^\/[a-z0-9]+\/(new|old[0-9]+)\// => "" }
    if (fileLocStr notin methods) { methods += (fileLocStr:()); }
    // map[str, map[str, list[map[str, value]]]] methods = (fileLocStr:());

    list[str] fileLines = readFileLines(fileLocation);

    Declaration ast = createAstFromFile(fileLocation, true);
    visit(ast) {
        case n:\method(_,name,_,_,_): {
            codeStr = flattenCode(fileLines, n.src);
            map[str, value] methodInfo = ("start_line":n.src.begin.line,
                                            "end_line":n.src.end.line,
                                            "code":codeStr,
                                            "node":n);

            if (name in methods[fileLocStr]) { methods[fileLocStr][name] += methodInfo; }
            else { methods[fileLocStr] += (name:[methodInfo]); }
        }
    }

    return methods;
}

// Create a list of all method locations in the file.
tuple[list[node], list[map[str, value]], list[map[str, value]]]
getMethodsNew(loc fileLocation,
                list[node] nodesNew,
                map[str, map[str, list[map[str, value]]]] methodsOldMap,
                map[str, map[str, value]] diffLines,
                map[str, value] baseInfo) {
    str fileLocStr = fileLocation.path;
    // Remove the commit specific folders from path
    fileLocStr = visit (fileLocStr) { case /^\/[a-z0-9]+\/(new|old[0-9]+)\// => "" }

    list[str] fileLines = readFileLines(fileLocation);

    list[map[str, value]] methodsNew = [];
    list[map[str, value]] methodsOld = [];
    Declaration ast = createAstFromFile(fileLocation, true);
    visit(ast) {
        case n:\method(_,name,_,_,_): {
            codeStr = flattenCode(fileLines, n.src);
            map[str, value] methodInfo = ("start_line":n.src.begin.line,
                                            "end_line":n.src.end.line,
                                            "code":codeStr,
                                            "node":n)
                                            + baseInfo;

            bool methodAdded = false;
            map[str, value] unchangedMethod = ();

            // compare the new method with the old commits.
            // If they are different, the method has been changed.
            <methodAdded, changedMethod, unchangedMethod> = writeMethod(name, methodInfo, methodsOldMap, diffLines, fileLocStr);

            if (methodAdded) { nodesNew += n; }
            if (unchangedMethod != ()) { methodsOld += unchangedMethod; }
            methodsNew += changedMethod;
        }
    }

    return <nodesNew, methodsNew, methodsOld>;
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
tuple[bool, map[str, value], map[str, value]]
writeMethod(str methodName,
            map[str, value] methodInfo,
            map[str, map[str, list[map[str, value]]]] methodsOldMap,
            map[str, map[str, value]] diffLines,
            str fileLocStr) {
    methodInfo += ("changed":false,
                    "path":fileLocStr,
                    "method_name":methodName,
                    "review":false,
                    "parent":false);
    
    // Marks exceptions in data
    if (diffLines[fileLocStr]["diff_before"] == "review") {
        // println("skipped due to review in diff");
        methodInfo["review"] = true;
        return <false, delete(methodInfo, "node"), ()>;
    }

    diffLinesList = typeCast(#map[str, list[list[int]]], diffLines[fileLocStr]);
    // A method is part of the change if it is new and thus cannot be found in the previous commit
    // If a file is added, the patch will show no lines before commit or [0,-1]
    list[list[int]] diffBefore = diffLinesList["diff_before"];
    if (fileLocStr notin methodsOldMap || methodName notin methodsOldMap[fileLocStr]) {
        methodInfo["changed"] = true;
        return <true, delete(methodInfo, "node"), ()>;
    }

    map[str, value] unchangedMethod = methodInfo;

    list[list[int]] diffAfter = diffLinesList["diff_after"];
    list[tuple[list[int], int]] diffAfterIndexed = zip2(diffAfter, [0..size(diffAfter)]);
    // Check if method has lines in any of the shown lines window of the patch
    for (<diff, idx> <- diffAfterIndexed) {
        if (diffCheck(methodInfo, diff)) {
            // Compare the new method with previous versions in the respective before diff
            for (oldInfo <- methodsOldMap[fileLocStr][methodName]) {
                list[int] diffBefore = diffLinesList["diff_before"][idx];
                node oldNode = typeCast(#node, oldInfo["node"]);
                if ((!(oldNode := methodInfo["node"])) && diffCheck(oldInfo, diffBefore)) {
                    methodInfo["changed"] = true;

                    // Add negative entry with unchanged code to set
                    unchangedMethod += ("code":oldInfo["code"],
                                        "start_line":diffBefore[0],
                                        "end_line":diffBefore[1],
                                        "parent":true,
                                        "changed":false,
                                        "node":oldNode);
                    return <true, delete(methodInfo, "node"), unchangedMethod>;
                }
            }
        }
    }
    // Add negative entry with unchanged code to set and filter entries with nodes that match any of the new_code.
    return <false, delete(methodInfo, "node"), unchangedMethod>;
    // Should exclude this method and only add the parent if possible,
    // bc otherwise you would get both parent and child as neg examples.
    // Or just always add new node to nodesNew in super function,
    // so that this method will always be added just once, be it parent or child.
}

// Compare the current method with previous versions to check if changes have been made
// map[str, value] methodChanged(str methodName,
//                               map[str, value] methodInfo,
//                               map[str, map[str, list[map[str, value]]]] methodsOld,
//                               map[str, map[str, value]] diffLines,
//                               str fileLocStr) {
//     methodInfo += ("changed":false, "old_code":"");
    
//     // iprintln(diffLines[fileLocStr]["diff_before"]);
//     if (diffLines[fileLocStr]["diff_before"] == "review") {
//         // println("skipped due to review in diff");
//         methodInfo["old_code"] = "review";
//         return delete(methodInfo, "node");
//     }

//     diffLinesList = typeCast(#map[str, list[list[int]]], diffLines[fileLocStr]);
//     // A method is part of the change if it is new and thus cannot be found in the previous commit
//     // If a file is added, the patch will show no lines before commit or [0,-1]
//     list[list[int]] diffBefore = diffLinesList["diff_before"];
//     if (fileLocStr notin methodsOld || methodName notin methodsOld[fileLocStr]) {
//         // if (fileLocStr notin methodsOld) { assert diffBefore == [] || diffBefore[0][1] == -1; }
//         methodInfo["changed"] = true;
//         return delete(methodInfo, "node");
//     }

//     list[list[int]] diffAfter = diffLinesList["diff_after"];
//     list[tuple[list[int], int]] diffAfterIndexed = zip2(diffAfter, [0..size(diffAfter)]);
//     // Check if method has lines in any of the shown lines window of the patch
//     for (diff <- diffAfterIndexed) {
//         if (diffCheck(methodInfo, diff[0])) {
//             // Compare the new method with previous versions in the respective before diff
//             for (oldInfo <- methodsOld[fileLocStr][methodName]) {
//                 list[int] diffBefore = diffLinesList["diff_before"][diff[1]];
//                 node oldNode = typeCast(#node, oldInfo["node"]);
//                 if ((!(oldNode := methodInfo["node"])) && diffCheck(oldInfo, diffBefore)) {
//                     methodInfo["changed"] = true;
//                     methodInfo["old_code"] = oldInfo["code"];
//                     return delete(methodInfo, "node");
//                 }
//             }
//         }
//     }
//     return delete(methodInfo, "node");
// }

bool diffCheck(map[str, value] methodInfo, list[int] diff) {
    return typeCast(#int, methodInfo["start_line"]) <= diff[1]
            && typeCast(#int, methodInfo["end_line"]) >= diff[0];
}

// Create a map containing all changed files from a commit and find which functions
// have been changed
// tuple[map[str, map[str, list[map[str, value]]]], set[str]] getChangedMethodsOld(loc commitLocation, set[str] new_code) {
//     set[loc] fileLocationsNew = files(commitLocation + "new");
//     if (fileLocationsNew == {}) { return (); }
//     set[loc] fileLocationsOld = files(commitLocation) - fileLocationsNew;

//     // Read the patch lines from the commit's json
//     loc diffLinesLoc = commitLocation + "diff_lines.json";
//     map[str, map[str, value]] diffLines =
//         readJSON(#map[str, map[str, value]], diffLinesLoc);
//     if (diffLines == ()) { return (); }

//     map[str, map[str, list[map[str, value]]]] methodsOld = ();
//     for (file <- fileLocationsOld) { methodsOld = getMethods(file, methodsOld, (), ()); }

//     map[str, map[str, list[map[str, value]]]] methodsNew = ();
//     for (file <- fileLocationsNew) { methodsNew = getMethods(file, methodsNew, methodsOld, diffLines); }

//     return <methodsNew, new_code>;
// }

// Create a map containing all changed files from a commit and find which functions
// have been changed
tuple[list[node], list[map[str, value]], list[map[str, value]]]
getChangedMethods(loc commitLocation,
                    list[node] nodesNew,
                    list[map[str, value]] methodsOld,
                    map[str, value] baseInfo) {
    set[loc] fileLocationsNew = files(commitLocation + "new");
    if (fileLocationsNew == {}) { return <nodesNew, [], []>; }
    set[loc] fileLocationsOld = files(commitLocation) - fileLocationsNew;

    // Read the patch lines from the commit's json
    loc diffLinesLoc = commitLocation + "diff_lines.json";
    map[str, map[str, value]] diffLines =
        readJSON(#map[str, map[str, value]], diffLinesLoc);
    if (diffLines == ()) { return <nodesNew, [], []>; }

    map[str, map[str, list[map[str, value]]]] methodsOldMap = ();
    for (file <- fileLocationsOld) { methodsOldMap = getMethodsOld(file, methodsOldMap); }

    list[map[str, value]] methodsNewAll = [];
    list[map[str, value]] methodsOldAll = [];
    list[map[str, value]] methodsNew = [];
    for (file <- fileLocationsNew) {
        <nodesNew, methodsNew, methodsOld> =
            getMethodsNew(file,
                            nodesNew,
                            methodsOldMap,
                            diffLines,
                            baseInfo);
        methodsNewAll += methodsNew;
        methodsOldAll += methodsOld;
    }

    return <nodesNew, methodsNewAll, methodsOldAll>;
}

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
import Exception;


/**
 * Create a list of all methods and their information in the old version of a file.
 *
 * @param fileLocation The location of the file to analyze.
 * @param methodsMap A map containing existing methods information.
 * @param baseInfo A map of base information to be added to each method's info.
 * @return A tuple containing an updated methods map and a list of method information maps.
 */
public tuple[map[str, map[str, list[map[str, value]]]], list[map[str, value]]]
getMethodsOld(loc fileLocation,
                map[str, map[str, list[map[str, value]]]] methodsMap,
                map[str, value] baseInfo) {
    str fileLocStr = fileLocation.path;
    // Remove the commit specific folders from path
    fileLocStr = visit (fileLocStr) { case /^(\/.+)*\/[a-z0-9]+\/old\// => "" }
    if (fileLocStr notin methodsMap) { methodsMap += (fileLocStr:()); }

    list[str] fileLines = readFileLines(fileLocation);

    list[map[str, value]] methods  = [];
    Declaration ast = createAstFromFile(fileLocation, true);
    visit(ast) {
        case n:\method(_,name,_,_,_): {
            codeStr = flattenCode(fileLines, n.src);
            map[str, value] methodInfo = ("path":fileLocStr,
                                            "method_name":name,
                                            "start_line":n.src.begin.line,
                                            "end_line":n.src.end.line,
                                            "code":codeStr,
                                            "parent":true,
                                            "changed":false,
                                            "node":n)
                                         + baseInfo;

            if (name in methodsMap[fileLocStr]) { methodsMap[fileLocStr][name] += methodInfo; }
            else { methodsMap[fileLocStr] += (name:[methodInfo]); }
            methods += methodInfo;
        }
    }

    return <methodsMap, methods>;
}


/**
 * Create a list of all methods and their information in the new version of a file.
 *
 * @param fileLocation The location of the file to analyze.
 * @param methodsOldMap A map containing methods information from the old commit.
 * @param diffLines A map containing the diff lines information.
 * @param baseInfo A map of base information to be added to each method's info.
 * @return A tuple containing a list of new method information maps.
 */
list[map[str, value]]
getMethodsNew(loc fileLocation,
                map[str, map[str, list[map[str, value]]]] methodsOldMap,
                map[str, map[str, value]] diffLines,
                map[str, value] baseInfo) {
    str fileLocStr = fileLocation.path;
    // Remove the commit specific folders from path
    fileLocStr = visit (fileLocStr) { case /^(\/.+)*\/[a-z0-9]+\/new\// => "" }

    list[str] fileLines = readFileLines(fileLocation);

    list[map[str, value]] methodsNew = [];
    Declaration ast = createAstFromFile(fileLocation, true);
    visit(ast) {
        case n:\method(_,name,_,_,_): {
            codeStr = flattenCode(fileLines, n.src);
            map[str, value] methodInfo = ("path":fileLocStr,
                                            "method_name":name,
                                            "start_line":n.src.begin.line,
                                            "end_line":n.src.end.line,
                                            "code":codeStr,
                                            "parent":false,
                                            "changed":true,
                                            "node":n)
                                         + baseInfo;

            // Compare the new method with the old commits.
            // If they are different, the method has been changed.
            map[str, value] newMethod = writeMethod(methodInfo, methodsOldMap, diffLines, fileLocStr);
            if (newMethod != ()) {
                methodsNew += newMethod;
            }
        }
    }

    return methodsNew;
}


/**
 * Convert the method's code into a single string.
 *
 * @param fileLines The lines of the file.
 * @param methodLocation The location of the method in the file.
 * @return The method's code as a single string.
 */
str flattenCode(list[str] fileLines, loc methodLocation) {
    list[str] methodOriginal =
        fileLines[methodLocation.begin.line-1..methodLocation.end.line];

    str method = "";
    for (line <- methodOriginal) { method = method + "\n" + line; }

    return method[1..];
}


/**
 * Compare the current method with previous versions to check if changes have been made.
 *
 * @param methodInfo Information about the current method.
 * @param methodsOldMap A map containing methods information from the old commit.
 * @param diffLines A map containing the diff lines information.
 * @param fileLocStr The file location in string format.
 * @return A map containing the method information if it has changed, otherwise an empty map.
 */
map[str, value] writeMethod(map[str, value] methodInfo,
                            map[str, map[str, list[map[str, value]]]] methodsOldMap,
                            map[str, map[str, value]] diffLines,
                            str fileLocStr) {
    map[str, value] diffLinesFile = diffLines[fileLocStr];
    // A method is part of the change if it is new and thus cannot be found in the previous commit
    str oldFileName = typeCast(#str, diffLinesFile["old_file"]);
    str methodName = typeCast(#str, methodInfo["method_name"]);

    if (oldFileName notin methodsOldMap || methodName notin methodsOldMap[oldFileName]) {
        return methodInfo;
    }

    list[list[int]] diffBefore = typeCast(#list[list[int]], diffLinesFile["diff_before"]);
    list[list[int]] diffAfter = typeCast(#list[list[int]], diffLinesFile["diff_after"]);
    list[tuple[list[int], list[int]]] diffsZipped = zip2(diffBefore, diffAfter);
    diffsZipped = [diff | diff <- diffsZipped, diffCheck(methodInfo, diff[1])];
    if (diffsZipped == []) { return (); }
    list[map[str, value]] oldMethods = methodsOldMap[oldFileName][methodName];
    for (<diffBeforeRange, _> <- diffsZipped) {
        for (oldInfo <- oldMethods) {
            if (diffCheck(oldInfo, diffBeforeRange)) {
                // One of the methods before commit is found to be the same as
                // the method after commit, so we cannot use it as a positive example.
                node oldNode = typeCast(#node, oldInfo["node"]);
                if (areSimilar(oldNode, methodInfo["node"])) { return (); }
                else { oldMethods -= oldInfo; }
            }
        }
    }

    // None of the old methods were found to be the same, so we can conclude that the new method
    // is changed in this commit.
    return methodInfo;
}


/**
 * Check if the method is within the range of diff lines.
 *
 * @param methodInfo Information about the method.
 * @param diff The range of diff lines.
 * @return True if the method is within the diff range, otherwise false.
 */
bool diffCheck(map[str, value] methodInfo, list[int] diff) {
    // If the window range > 1, then the window includes removed/added lines
    // and only those should be checked.
    // Otherwise, the lines directly above and below are used,
    // as those windows do not contain any lines to check.
    int startLine = diff[0];
    int endLine = diff[1];
    if (endLine - startLine > 1) {
        startLine += 1;
        endLine -= 1;
    }
    return typeCast(#int, methodInfo["start_line"]) <= endLine
            && typeCast(#int, methodInfo["end_line"]) >= startLine;
}


/**
 * Compares two subtrees to determine if they are structurally similar.
 *
 * @param n1 The root of the first subtree to be compared.
 * @param n2 The root of the second subtree to be compared.
 * @return `true` if the nodes are structurally similar after unsetting 
 *         specified attributes, `false` otherwise.
 */
public bool areSimilar(node n1, node n2) {
    n1 = unsetRec(n1, {"src", "decl"});
    n2 = unsetRec(n2, {"src", "decl"});

    list[node] tree1 = [];
    list[node] tree2 = [];

    visit (n1) { case node n: tree1 += n; }
    visit (n2) { case node n: tree2 += n; }

    return tree1 == tree2;
}


/**
 * Create a list containing all methods changed in a commit.
 *
 * @param commitLocation The location of the commit to analyze.
 * @param baseInfo A map of base information to be added to each method's info.
 * @return A tuple containing a list of new method information maps, and a list of old method information maps.
 */
public tuple[list[map[str, value]], list[map[str, value]]]
getChangedMethods(loc commitLocation, map[str, value] baseInfo) {
    set[loc] fileLocationsNew = files(commitLocation + "new");
    if (fileLocationsNew == {}) { return <[], []>; }
    set[loc] fileLocationsOld = files(commitLocation) - fileLocationsNew;

    // Read the patch lines from the commit's json
    value readDiffLines(loc l) {
        try return readJSON(#str, l);
        catch Java("IllegalStateException",
                    "Expected a string but was BEGIN_OBJECT at line 1 column 2 path $"): {
            return readJSON(#map[str, map[str, value]], l);
        }
    }
    diffLinesCheck = readDiffLines(commitLocation + "diff_lines.json");
    if (typeOf(diffLinesCheck) == \str() || diffLinesCheck == ()) { return <[], []>; }
    map[str, map[str, value]] diffLines = typeCast(#map[str, map[str, value]], diffLinesCheck);

    map[str, str] fileLocsPaired = (typeCast(#str, diffLines[newFile]["old_file"]) : newFile
                                    | newFile <- diffLines,
                                      diffLines[newFile]["old_file"] != "/dev/null");

    // Process each file in the old commit to extract methods
    map[str, map[str, list[map[str, value]]]] methodsOldMap = ();
    list[map[str, value]] methodsOldAll = [];
    println("Total files: <size(fileLocationsOld) + size(fileLocationsNew)>");
    int i = 0;
    for (file <- fileLocationsOld) {
        if (i%20 == 0) { println("Processing file #<i>"); }
        <methodsOldMap, methodsOld> = getMethodsOld(file,
                                                    methodsOldMap,
                                                    baseInfo);
        methodsOldAll += methodsOld;
        i += 1;
    }

    // Process each file in the new commit to extract and compare methods
    list[map[str, value]] methodsNewAll = [];
    for (file <- fileLocationsNew) {
        if (i%20 == 0) { println("Processing file #<i>"); }
        methodsNew = getMethodsNew(file,
                                    methodsOldMap,
                                    diffLines,
                                    baseInfo);
        methodsNewAll += methodsNew;
        i += 1;
    }

    return <methodsNewAll, methodsOldAll>;
}

module Extract

import IO;
import util::FileSystem;
import lang::java::m3::Core;
import lang::java::m3::AST;
import lang::json::IO;
import List;
import String;

// Create a list of all method locations in the file.
public map[loc, map[str, list[map[str, value]]]]
getMethods(loc fileLocation) {
    map[loc, map[str, list[map[str, value]]]] methods = (fileLocation:());
    list[str] fileLines = readFileLines(fileLocation);

    Declaration ast = createAstFromFile(fileLocation, true);
    visit(ast) {
        case a:\method(_,name,_,_,_): {
            map[str, value] methodInfo = ("start_line":a.src.begin.line,
                                            "end_line":a.src.end.line,
                                            "code":flattenCode(fileLines, a.src));
            if (name in methods[fileLocation]) { methods[fileLocation][name] += methodInfo; }
            else { methods[fileLocation] += (name:[methodInfo]); }
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

// void getChangedMethods(loc changesLocation) {
//     // list[str] fileLocations = readJSON(#list[str], changesLocation);
//     set[loc] fileLocationsNew = files(changesLocation + "new");
//     map[loc, list[map[str, value]]] methodsNew = ();

//     for (file <- fileLocationsNew) { methodsNew += getMethods(file); }

//     set[loc] fileLocationsOld = files(changesLocation) - fileLocationsNew;
//     map[loc, list[map[str, value]]] methodsOld = ();
//     for (file <- fileLocationsOld) {
//         // str locStr = file.path;
//         methodsOld += getMethods(file);
//     }

    // TODO: Need to compare new files to old files
    // If function is found to be changed, go next
// }

// Get the diff functions from two different commits
// void getDiffFunctions(loc newLocation, loc oldLocation) {
    
// }
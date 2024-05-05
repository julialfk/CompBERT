module Extract

import IO;
import util::FileSystem;
import lang::java::m3::Core;
import lang::java::m3::AST;
import List;
import String;

// Create a list of all method locations in the file.
public list[map[str, value]] getMethods(loc fileLocation) {
    list[map[str, value]] methods = [];
    list[str] fileLines = readFileLines(fileLocation);

    Declaration ast = createAstFromFile(fileLocation, true);
    visit(ast) {
        case a:\method(_,name,_,_,_): {
            methods += ("func_name":name,
                        "path":a.src.path,
                        "start_line":a.src.begin.line,
                        "end_line":a.src.end.line,
                        "code":flattenCode(fileLines, a.src));
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
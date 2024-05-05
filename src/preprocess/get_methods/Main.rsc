module Main

import IO;
import lang::java::m3::Core;
import lang::java::m3::AST;
import lang::json::IO;
import List;
// import Set;
// import String;
import Extract;

// Run the analysis over multiple Java programs.
int mainList(list[loc] projectlocations) {
    for(project <- projectlocations) {
        main(project);
    }

    return 0;
}

// Create a list of all file locations in the project.
list[loc] getFiles(loc projectLocation) {
    M3 model = createM3FromMavenProject(projectLocation);
    list[loc] fileLocations = [f | f <- files(model.containment), isCompilationUnit(f)];
    return fileLocations;
}

// 
int main(loc projectLocation) {
    list[loc] fileLocations = getFiles(projectLocation);
    list[map[str, value]] methods = [];

    for(file <- fileLocations) {
        methods += getMethods(file);
    }

    writeJSON(|project://preprocess/output/iTrust.json|, methods);

    return 0;
}
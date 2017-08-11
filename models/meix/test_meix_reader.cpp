#include <string>
#include <list>
#include <vector>
#include <iostream>
#include "meix2.h"


main () {
    std:: string input_file="/home/mmarin/Desktop/Limb_project_data/vtk_limbReferences/277_AER.vtk";

    Meix limb(input_file);
    std::cout<<"READ DONE"<<std::endl;


    //find a facet that is on the flank border (that is normal=-1,0,0)
    // int v;
    // for (int i=0 ; i<limb.n_facets ; i++) {
    //     if(limb.Facets[i].N.x==-1.f && limb.Facets[i].N.y==0.f && limb.Facets[i].N.z==0.f){
    //         v=i;
    //         break;
    //     }
    // }

    // for (int i=0 ; i<limb.n_vertices ; i++) {
    //     std::cout<<"pre "<<i<<" "<<limb.Vertices[i].x<<std::endl;
    // }


    // std::cout<<"old x of v "<<limb.Vertices[v].x<<std::endl;
    // Point translation_vector(-limb.Vertices[v].x, 0.f, 0.f);
    // limb.Translate(translation_vector);
    // std::cout<<"new x of v "<<limb.Vertices[v].x<<std::endl;

    // for (int i=0 ; i<limb.n_facets ; i++){
    //     std::cout<<"triangle "<<i<<" : "<<limb.triangle_to_vertices[i][0]<<" "<<limb.triangle_to_vertices[i][1]<<" "<<limb.triangle_to_vertices[i][2]<<std::endl;
    // }

    // for (int i=0 ; i<limb.n_facets ; i++){
    //     std::cout<<"Normal "<<limb.Facets[i].N.x<<" "<<limb.Facets[i].N.y<<" "<<limb.Facets[i].N.z<<std::endl;
    // }

    //print the list of triangles per vertex
    // for (int i=0 ; i<limb.n_vertices ; i++){
    //     std::cout<<"vertex "<<i<<" : ";
    //     for (int j=0 ; j<limb.vertex_to_triangles[i].size(); j++) {
    //         std::cout<<limb.vertex_to_triangles[i][j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    // limb.Rescale_relative(0.5f);
    limb.Rescale_absolute(50.f);

    for (int i=0 ; i<limb.n_facets ; i++) {
        std::cout<<i<<" "<<limb.Facets[i].C.x<<" "<<limb.Facets[i].N.x<<std::endl;
    }

    limb.WriteVtk("meix2test");
    std::cout<<"DONE"<<std::endl;
}

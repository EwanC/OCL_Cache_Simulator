

struct Node {
  int starting;
  int no_of_edges;
};


__kernel void BFS_kernel( const __global struct Node* g_graph_nodes,const __global int* g_graph_edges, __global bool* g_graph_mask, __global bool* g_updating_graph_mask, __global bool* g_graph_visited, __global int* g_cost, __global bool* g_over, const int no_of_nodes){

  int tid = get_global_id(0);
  
  if( tid<no_of_nodes && g_graph_mask[tid]){
     g_graph_mask[tid]=false;
     for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++){
      
       int id = g_graph_edges[i];
       if(!g_graph_visited[id]){
           g_cost[id]=g_cost[tid]+1;
           g_updating_graph_mask[id]=true;
       }  
      }
  } 

(CLK_GLOBAL_MEM_FENCE);
 
 if( tid<no_of_nodes && g_updating_graph_mask[tid]){
   g_graph_mask[tid]=true;
   g_graph_visited[tid]=true;
   *g_over=true;
   g_updating_graph_mask[tid]=false;
 }


}




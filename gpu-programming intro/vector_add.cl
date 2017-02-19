__kernel void vecAdd(  __global double *a,                       
                       __global double *b,                       
                       __global double *c,                       
                       const unsigned int n)                    
{                                                               
    //Get our global thread ID                                  
    int id = get_global_id(0);                                  
                                                                
    if (id < n)                                                 
        c[id] = a[id] + b[id];                                  
}
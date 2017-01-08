#include "ConvexSegment.h"
#include <armadillo>
#include <cmath>

using namespace arma;

namespace{

    void GCSegment_2phase_core_iteration(mat & segment_result, mat & u, mat & d_x, mat & d_y, mat & b_x, mat & b_y, mat & r);
    void GCSegment_2phase_mem_alloc(int row, int col, mat & u, mat & d_x, mat & d_y, mat & b_x, mat & b_y, mat & r);
}

void GCSegment_2phase_mem_alloc(int row, int col, mat & u, mat & d_x, mat & d_y, mat & b_x, mat & b_y, mat & r){
    u.zeros(row,col);
    d_x.zeros(row,col);
    d_y.zeros(row,col);
    b_x.zeros(row,col);
    b_y.zeros(row,col);
    r.zeros(row,col);
}

void GCSegment_2phase_core_iteration(mat & u, mat & d_x, mat & d_y, mat & b_x, mat & b_y, mat & r){
    // ***	The l2-penalty parameter (Lagrange multiplier)
    double lambda = 0.5;
    // ***	The data fidelity parameter 
    double mu = 10.0;
    
    int row = u.n_rows;
    int col = u.n_cols;
    
    double rel_error = 1.0;
    double tol = 1e-5;
    int num_iterations = 0;
    double max_new = 0.0, max_old;
    double alpha, beta, ux, uy, temp;
    
    while(rel_error > tol){
	max_old = max_new;
	max_new = 0.0;
	rel_error = 0.0;
	
	for(int x = 1; x < row-1; x++){
	    for(int y = 1; y < col-1; y++){
		ux = u(x+1,y) - u(x,y);
		alpha = ux + b_x(x,y);
		uy = u(x,y+1) - u(x,y);
		beta =  uy + b_y(x,y);
		d_x(x,y) = (fabs(alpha) == 0 ? 0.0 : alpha/fabs(alpha)*fmax(fabs(alpha)-1/lambda,0.0));
		d_y(x,y) = (fabs(beta) == 0 ? 0.0 : beta/fabs(beta)*fmax(fabs(beta)-1/lambda,0.0));
		b_x(x,y) = alpha - d_x(x,y);
		b_y(x,y) = beta - d_y(x,y);
	    }
	}

	for(int x = 1; x < row-1; x++){
	    for(int y = 1; y < col-1; y++){
		temp = b_x(x,y)-b_x(x-1,y)-d_x(x,y)+d_x(x-1,y);
		temp += b_y(x,y)-b_y(x,y-1)-d_y(x,y)+d_y(x,y-1);
		temp = (u(x-1,y) + u(x+1,y) + u(x,y-1) + u(x,y+1) + mu/lambda*r(x,y) + temp)/4.0;
		if(temp > 1)
		    temp = 1.0;
		else if (temp <0)
		    temp = 0.0;
		
		rel_error += pow(u(x,y)-temp, 2);
		max_new += temp;
		u(x,y) = temp;
	    }

	    u(x, col-1) = u(x, col-2);

	    //temporary fix
	    u(x, 0) = u(x, 1);
	}

	for (int j = 1; j < col-1; j++){
	    u(row-1, j) = u(row-2, j);
	    
	    //temporary fix
	    u(0, j) = u(1, j);
	}

	//temporary fix
	u(0,0) = u(1,0);
	u(row-1,0) = u(row-2,0);
	u(0,col-1) = u(1,col-1);
	u(row-1,col-1) = u(row-2,col-1);

	rel_error = rel_error/fmax(max_new, max_old);
	num_iterations++;
    }

    cout << "Convex ADMM Segmant, Iterations: " << num_iterations << ", Error Tolerance: " << tol << endl;
}



void GCSegment_1c(mat & image, mat & segments, double channel_averages[2]){

    int row = image.n_rows;
    int col = image.n_cols;
    
    mat u(image);
    mat b_x(row, col, fill::zeros);
    mat b_y(row, col, fill::zeros);
    mat d_x(row, col, fill::zeros);
    mat d_y(row, col, fill::zeros);
    mat r(row, col, fill::zeros);
    
    // GCSegment_2phase_mem_alloc(row, col, u, d_x, d_y, b_x, b_y, r);
    
    for (int x = 0; x < row; x++)
	for (int y = 0; y < col; y++)
	    r(x,y) = pow(image(x,y) - channel_averages[0], 2) - pow(image(x,y) - channel_averages[1], 2);
    
    GCSegment_2phase_core_iteration(u, d_x, d_y, b_x, b_y, r);
    
    segments = u;
}


void GCS_threshold(mat & raw_segments, double threshold){

    int row = raw_segments.n_rows;
    int col = raw_segments.n_cols;
    
    for(int x = 0; x < row; x++){
	for(int y = 0; y < col; y++){
	    raw_segments(x,y) = (raw_segments(x,y) > threshold) ? 1.0 : 0.0;
	}
    }
}

void GCS_segments_to_image(mat & raw_segments, mat & image_result, double channel_averages[2]){

    int row = raw_segments.n_rows;
    int col = raw_segments.n_cols;
    
    mat true_segments = raw_segments;
    GCS_threshold(true_segments,0.5);
    imat isegments = conv_to<imat>::from(true_segments);

    for (int x = 0; x < row; x++){
	for (int y = 0; y < col; y++){
	    image_result(x,y) = channel_averages[isegments(x,y)];
	}
    }
}

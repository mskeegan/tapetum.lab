#include "ConvexSegment.h"

#include "ADMMDenoise.h"
#include <armadillo>
#include <cmath>
#include <vector>

using namespace arma;
using namespace std;

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

    cout << "Convex ADMM Segment, Iterations: " << num_iterations << ", Error Tolerance: " << tol << endl;
}

void GCSegment_1c(const mat & image, mat & segments, double channel_averages[2]){

    int row = image.n_rows;
    int col = image.n_cols;
    
    mat u(image);
    mat b_x(row, col, fill::zeros);
    mat b_y(row, col, fill::zeros);
    mat d_x(row, col, fill::zeros);
    mat d_y(row, col, fill::zeros);
    mat r(row, col, fill::zeros);
    
    //GCSegment_2phase_mem_alloc(row, col, u, d_x, d_y, b_x, b_y, r);
    
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

void GCS_threshold(const vector<mat> & raw_segments, umat & labels){

    int segments = raw_segments.size();
    int row = raw_segments[0].n_rows;
    int col = raw_segments[0].n_cols;
    //labels.zeros();

    /*cube segment_mat(row,col,segments,fill::zeros);
    for (int j = 0; j < segments; j++){
	segment_mat.slice(j) = raw_segments[j];
	}*/

    double max_val;
    uint max_index;
    
    for (int y = 0; y < col; y++){
	for (int x = 0; x < row; x++){
	    max_val = 0.0;
	    max_index = 0;
	    for (int j = 0; j < segments; j++){
		if (raw_segments[j](x,y) > max_val){
		    max_index = j;
		    max_val = raw_segments[j](x,y);
		}
	    }
	    labels(x,y) = max_index;
	}
    }
}

void GCS_segments_to_image(const mat & raw_segments, mat & image_result, double channel_averages[2]){

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

void GCS_segments_to_image(vector<mat> & segments, mat & image_result, double* channel_averages){

    int clusters = segments.size();
    int row = segments[0].n_rows;
    int col = segments[0].n_cols;

    // Using copy constructor to copy each mat 
    umat thresholded_segments(row,col,fill::zeros);
    cout << "Labels extrema: (" << thresholded_segments.min() << "," << thresholded_segments.max() << ")" << endl;
    GCS_threshold(segments,thresholded_segments);

    //#ifdef IMAGING_DEBUG
    cout << "Constructing quantized image. " << "Clusters: " << clusters << endl;
    cout << "Labels dimensions: (" << thresholded_segments.n_rows << "," << thresholded_segments.n_cols << ")" << endl;
    cout << "Labels extrema: (" << thresholded_segments.min() << "," << thresholded_segments.max() << ")" << endl;
    //#endif

    for (int x = 0; x < row; x++){
	for (int y = 0; y < col; y++){
	    image_result(x,y) = channel_averages[thresholded_segments(x,y)];
	}
    }
}

void Simplex_projection(vec & c){
    
	vec point = c;
	int rows = c.n_rows;
	uvec indices(rows,fill::zeros); 
	double point_sum;
	int point_dim = rows;

	bool continue_iteration = true;
	int iterations = 0;
	while(continue_iteration){
		point_sum = 0;
		continue_iteration = false;
		for(int x = 0; x < rows; x++)
			point_sum += point(x);
		for(int x = 0; x < rows; x++){
			if(indices(x) == 0){
				point(x) = point(x) - (point_sum-1.0)/point_dim;
				if(point(x) < 0){
					continue_iteration = true;
					point(x) = 0;
					indices(x) = 1;
				}
			}
		}
		point_dim = 0;
		for(int x = 0; x < rows; x++)
			if(indices(x) == 0)
				point_dim++;

		iterations++;

#ifdef IMAGING_DEBUG_DEEP
		cout << "(DEBUG)Iteration " << iterations << endl;
		cout << "(DEBUG)Point: " << point << endl;
		cout << "(DEBUG)Indices: " << indices << endl;
#endif
	}

	c = point;

#ifdef IMAGING_DEBUG_DEEP
	cout << "(DEBUG)Simplex Project Iterations: " << iterations << endl;
#endif
}

void GCSegment_SplitBregman_Multiphase(const mat & image, vector<mat> & segments, int num_channels, double * channel_averages, double lambda){
	int num_rows = image.n_rows;
	int num_cols = image.n_cols;

	vector<mat> u(num_channels);
	vector<mat> v(num_channels);
	vector<mat> b_x(num_channels);
	vector<mat> b_y(num_channels);
	for(int l = 0; l < num_channels; l++){
	    u[l] = zeros(num_rows,num_cols);
	    v[l] = zeros(num_rows,num_cols);
	    b_x[l] = zeros(num_rows,num_cols);
	    b_y[l] = zeros(num_rows,num_cols);
	}
	mat d_x_tempholder(num_rows,num_cols,fill::zeros);
	mat d_y_tempholder(num_rows,num_cols,fill::zeros);


        // initialize channel penalties - currently manually
        // TODO: I need to implement a k-means or let the use pass in penalties
        //double colors[4] = {0.9, 0.6, 0.3, 0.0};
	vector<mat> channels(num_channels);
	for(int j = 0; j < num_channels; j++)
		channels[j] = zeros(num_rows,num_cols);

	for(int j = 0; j < num_channels; j++){ 
		for(int x = 0; x < num_rows; x++){
			for(int y = 0; y < num_cols; y++){
				channels[j](x,y) = pow(image(x,y) - channel_averages[j],2); 		
			}
		}
	}
	//if(seg_result.getDataPointer() == NULL)
	//seg_result.initialize(num_rows, num_cols);
	
	vec vectosimplex(num_channels,fill::zeros);
	double theta = 10.0;
	int max_outer_iterations = 150;
	double tol = 0.0;

	// ***	Initializing v variables
	for(int x = 0; x < num_rows; x++){
		for(int y = 0; y < num_cols; y++){
		        for(int j = 0; j < num_channels; j++){
				vectosimplex(j) = (u[j](x,y) - lambda*theta*channels[j](x,y));
		        }
			Simplex_projection(vectosimplex);
			for(int j = 0; j < num_channels; j++)
				v[j](x,y) = vectosimplex(j);
		}
	}

	// ***	Main Iterations
	int k;
	for(k = 0; k < max_outer_iterations; k++){
		// ***	Compute Gradient: TV-L2 Solve
		for(int j = 0; j < num_channels; j++){
			ad_core_iteration(v[j], u[j], b_x[j], b_y[j], d_x_tempholder, d_y_tempholder, 1.0/2*theta);
		}

		// ***	Project Onto Simplex
		for(int x = 0; x < num_rows; x++){
			for(int y = 0; y < num_cols; y++){
				for(int j = 0; j < num_channels; j++)
					vectosimplex(j) = (u[j](x,y) - lambda*theta*channels[j](x,y));
				Simplex_projection(vectosimplex);
				for(int j = 0; j < num_channels; j++)
					v[j](x,y) = vectosimplex(j);
			}
		}

#ifdef IMAGING_PROTOTYPING
		mat seg_result(num_rows,num_cols,fill::zeros);
		// ***	Computing Intermediate\Final Segmentation Result	
		for(int x = 0; x < num_rows; x++){
			for(int y = 0; y < num_cols; y++){
				seg_result(x,y) = 0;

				for(int j = 1; j < num_channels; j++){ 
					if(v[j](x,y) > v[(int)seg_result(x,y)](x,y))
						seg_result(x,y) = j;
				}
			}
		}
		
		char s[30];
		sprintf(s, "SBMPtest%d.bmp", k+1);
		string outfile(s);
		string fullfile = filepath + "ImageResults/ConvexSegmentationResults/" + outfile;
		writeBMPFile(seg_result/num_channels, fullfile);
#endif
	}

	segments = v;

	cout << "Convex ADMM Multiphase Segment, Iterations: " << k << ", Error Tolerance: " << tol << endl;
}

/*
 * alg_comm.h
 *
 *  Created on: Dec 27, 2022
 *      Author: flyb
 */

#ifndef ALG_ALG_COMM_H_
#define ALG_ALG_COMM_H_
#include "rknn_api.h"

extern unsigned char* load_model(const char *filename, int *model_size);
extern  void dump_tensor_attr(rknn_tensor_attr *attr);


#endif /* ALG_ALG_COMM_H_ */

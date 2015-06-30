/**********************************************************************
  quick_sort.h - sorts a vector of type std::vector<T> using a quick-sort
  algorithm.
  Copyright (C) 2015 by Patrick Avery
  This source code is released under the New BSD License, (the "License").
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
***********************************************************************/


#ifndef QUICK_SORT_H
#define QUICK_SORT_H

#include <vector>

class quick_sort
{
 public:
  template<typename T> static void quickSort(std::vector<T> &list,
                                             int start = 0, int end = 0);
};

#include "quick_sort.tpp"

#endif

/**********************************************************************
  quick_sort.tpp - sorts a vector of type std::vector<T> using a quick-sort
  algorithm.
  Copyright (C) 2015 by Patrick Avery
  This source code is released under the New BSD License, (the "License").
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
***********************************************************************/

#include <algorithm>

#include "quick_sort.h"

template<typename T> void quick_sort::quickSort(std::vector<T> &list,
                                                int start, int end)
{
  // Set the end to the list size if the end is zero (default parameter).
  if (end == 0) end = list.size() - 1;

  // comparisonIndex keeps track of the position of the element that we are
  // comparing with the element at the end index.
  // swapIndex keeps track of the position of the element to be swapped
  // with comparisonIndex if the element at comparisonIndex is found to be
  // less than the element at the end position.
  // This essentially moves all elements smaller than the element at the end
  // index to the left side.
  int comparisonIndex = start;
  int swapIndex = start;
  while (comparisonIndex != end) {
    if (list[comparisonIndex] <= list[end]) {
      std::swap(list[comparisonIndex], list[swapIndex]);
      // Increment the swapIndex if an item is swapped.
      swapIndex++;
    }
    // Always increment the comparisonIndex after each iteration.
    comparisonIndex++;
  }
  // Place the element at the end index to the right of all the elements that
  // it is greater than.
  std::swap(list[swapIndex], list[end]);

  if (start < swapIndex - 1) {
    // If the block size is greater than 1000, parallelize it. Otherwise, don't
    if (swapIndex - 1 - start > 1000) {
      #pragma omp task shared(list)
      {
        quick_sort::quickSort(list, start, swapIndex - 1);
      }
    }
    else quick_sort::quickSort(list, start, swapIndex - 1);
  }
  if (swapIndex + 1 < end) {
    // If the block size is greater than 1000, parallelize it. Otherwise, don't
    if (end - swapIndex - 1 > 1000) {
      #pragma omp task shared(list)
      {
        quick_sort::quickSort(list, swapIndex + 1, end);
      }
    }
    else quick_sort::quickSort(list, swapIndex + 1, end);
  }
  #pragma omp taskwait
}


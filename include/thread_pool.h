#pragma once

#include <functional>
#include <thread>
#include <vector>

namespace mini_llama {

int GetThreadCount();

void SetThreadCount(int n);

void ParallelFor(int n, const std::function<void(int begin, int end)>& work);

}
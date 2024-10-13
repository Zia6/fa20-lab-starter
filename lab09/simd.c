#include <time.h>
#include <stdio.h>
#include <x86intrin.h>
#include "simd.h"

long long int sum(int vals[NUM_ELEMS])
{
	clock_t start = clock();

	long long int sum = 0;
	for (unsigned int w = 0; w < OUTER_ITERATIONS; w++)
	{
		for (unsigned int i = 0; i < NUM_ELEMS; i++)
		{
			if (vals[i] >= 128)
			{
				sum += vals[i];
			}
		}
	}
	clock_t end = clock();
	printf("Time taken: %Lf s\n", (long double)(end - start) / CLOCKS_PER_SEC);
	return sum;
}

long long int sum_unrolled(int vals[NUM_ELEMS])
{
	clock_t start = clock();
	long long int sum = 0;

	for (unsigned int w = 0; w < OUTER_ITERATIONS; w++)
	{
		for (unsigned int i = 0; i < NUM_ELEMS / 4 * 4; i += 4)
		{
			if (vals[i] >= 128)
				sum += vals[i];
			if (vals[i + 1] >= 128)
				sum += vals[i + 1];
			if (vals[i + 2] >= 128)
				sum += vals[i + 2];
			if (vals[i + 3] >= 128)
				sum += vals[i + 3];
		}

		// This is what we call the TAIL CASE
		// For when NUM_ELEMS isn't a multiple of 4
		// NONTRIVIAL FACT: NUM_ELEMS / 4 * 4 is the largest multiple of 4 less than NUM_ELEMS
		for (unsigned int i = NUM_ELEMS / 4 * 4; i < NUM_ELEMS; i++)
		{
			if (vals[i] >= 128)
			{
				sum += vals[i];
			}
		}
	}
	clock_t end = clock();
	printf("Time taken: %Lf s\n", (long double)(end - start) / CLOCKS_PER_SEC);
	return sum;
}

long long int sum_simd(int vals[NUM_ELEMS])
{
	clock_t start = clock();
	__m128i _127 = _mm_set1_epi32(127); // Set threshold 127
	long long int result = 0;			// Use 64-bit integer for result

	// 避免在每次迭代中存储，减少存储操作的频率
	for (unsigned int w = 0; w < OUTER_ITERATIONS; w++)
	{
		__m128i _sum = _mm_setzero_si128(); // Initialize SIMD sum accumulator
		for (unsigned int i = 0; i < NUM_ELEMS / 4 * 4; i += 4)
		{
			__m128i _val = _mm_loadu_si128((__m128i *)(vals + i)); // Load 4 values
			__m128i _cmp = _mm_cmpgt_epi32(_val, _127);			   // Compare with 127
			_val = _mm_and_si128(_val, _cmp);					   // Mask values <= 127
			_sum = _mm_add_epi32(_sum, _val);					   // Accumulate the sum
		}
		// 提取最后一次未处理的值
		unsigned int rel[4];
		_mm_storeu_si128((__m128i *)rel, _sum);
		result += rel[0] + rel[1] + rel[2] + rel[3];

		// 处理不能被4整除的剩余元素
		for (unsigned int i = NUM_ELEMS / 4 * 4; i < NUM_ELEMS; i++)
		{
			if (vals[i] >= 128)
			{
				result += vals[i];
			}
		}
	}

	clock_t end = clock();
	printf("Time taken: %Lf s\n", (long double)(end - start) / CLOCKS_PER_SEC);
	return result;
}

long long int sum_simd_unrolled(int vals[NUM_ELEMS])
{
	clock_t start = clock();
	__m128i _127 = _mm_set1_epi32(127); // Set threshold 127
	long long int result = 0;			// Use 64-bit integer for final result

	for (unsigned int w = 0; w < OUTER_ITERATIONS; w++)
	{
		__m128i _sum1 = _mm_setzero_si128(); // SIMD sum accumulator 1
		// 每次处理 16 个元素（循环展开）
		for (unsigned int i = 0; i < NUM_ELEMS / 16 * 16; i += 16)
		{
			// 加载 16 个值到四个 SIMD 寄存器
			__m128i _val1 = _mm_loadu_si128((__m128i *)(vals + i));		 // Load first 4 values
			__m128i _val2 = _mm_loadu_si128((__m128i *)(vals + i + 4));	 // Load next 4 values
			__m128i _val3 = _mm_loadu_si128((__m128i *)(vals + i + 8));	 // Load next 4 values
			__m128i _val4 = _mm_loadu_si128((__m128i *)(vals + i + 12)); // Load next 4 values

			// 与 127 进行比较，保留大于 127 的值
			__m128i _cmp1 = _mm_cmpgt_epi32(_val1, _127); // Compare first 4 values
			__m128i _cmp2 = _mm_cmpgt_epi32(_val2, _127); // Compare next 4 values
			__m128i _cmp3 = _mm_cmpgt_epi32(_val3, _127); // Compare next 4 values
			__m128i _cmp4 = _mm_cmpgt_epi32(_val4, _127); // Compare next 4 values
			_val1 = _mm_and_si128(_val1, _cmp1);		  // Mask first 4 values
			_val2 = _mm_and_si128(_val2, _cmp2);		  // Mask next 4 values
			_val3 = _mm_and_si128(_val3, _cmp3);		  // Mask next 4 values
			_val4 = _mm_and_si128(_val4, _cmp4);		  // Mask next 4 values

			// 将四个 SIMD 向量的结果分别累加到四个累加寄存器中
			_sum1 = _mm_add_epi32(_sum1, _val1);
			_sum1 = _mm_add_epi32(_sum1, _val2);
			_sum1 = _mm_add_epi32(_sum1, _val3);
			_sum1 = _mm_add_epi32(_sum1, _val4);
		}
		// 提取最后一次未处理的值
		unsigned int rel[4];
		_mm_storeu_si128((__m128i *)rel, _sum1);
		result += rel[0] + rel[1] + rel[2] + rel[3];
		// 处理不能被 16 整除的剩余元素
		for (unsigned int i = NUM_ELEMS / 16 * 16; i < NUM_ELEMS; i++)
		{
			if (vals[i] >= 128)
			{
				result += vals[i];
			}
		}
	}

	clock_t end = clock();
	printf("Time taken: %Lf s\n", (long double)(end - start) / CLOCKS_PER_SEC);
	return result;
}

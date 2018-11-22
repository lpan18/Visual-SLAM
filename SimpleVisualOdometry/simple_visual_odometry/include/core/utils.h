#define EIGEN_DONT_ALIGN_STATICALLY
#ifndef INCLUDE_UTILS_H_
#define INCLUDE_UTILS_H_

#include <vector>

inline std::ostream& operator<<(std::ostream& os,
			const std::vector<unsigned char>& v)
{
	if (!v.empty())
	{
		size_t i;
		for (i = 0; i + 1 < v.size(); i++)
			os << (v[i] != 0) << ",";

		os << v[i];
	}
	return os;
}

inline unsigned int countInlier(const std::vector<unsigned char>& v)
{
	unsigned int count = 0;
	for(auto c : v)
	{
		if(c != 0)
			count++;
	}

	return count;

}

#endif /* INCLUDE_UTILS_H_ */

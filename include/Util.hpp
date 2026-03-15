#pragma once

#include <chrono>
#include <numeric>
using namespace std::chrono;
#include <stdexcept>
#include <vulkan/vulkan.hpp>
#include <optional>
#include <expected>
#include <array>
#include <cassert>
// #include <unordered_map>
// #include <print>

template<class T>
inline T* Unwrap(T* res, const char *msg) {
	if (res != nullptr) return res;
	else throw std::runtime_error(msg);
}
template<class T>
inline T Unwrap(vk::ResultValue<T> res, const char *msg) {
	if (res.result == vk::Result::eSuccess) return res.value;
	else throw std::runtime_error(msg);
}
template<class T>
inline T Unwrap(std::optional<T> res, const char *msg) {
	if (res.has_value()) return res.value();
	else throw std::runtime_error(msg);
}
template<class T, class S>
inline T Unwrap(std::expected<T, S> res) {
	if (res.has_value()) return res.value();
	else throw std::runtime_error(res.error());
}
inline void Unwrap(vk::Result res, const char *msg) {
	if (res == vk::Result::eSuccess) return;
	else throw std::runtime_error(msg);
}

class Timer {
	std::optional<steady_clock::time_point> m_Start;

	public:
	void start() {
		m_Start = steady_clock::now();
	}
	template <typename Duration = milliseconds>
	steady_clock::rep stop() {
		assert(m_Start.has_value());
		auto time = duration_cast<Duration>(steady_clock::now() - *m_Start).count();
		m_Start.reset();
		return time;
	}
};

template <int Count = 20, class Duration = milliseconds>
class AverageTimer {
	std::array<steady_clock::rep, Count> m_Times{0};
	int m_CurrentIndex{};
	int m_Count{};
	std::optional<steady_clock::time_point> m_Start;

	public:
	void start() {
		m_Start = steady_clock::now();
	};
	steady_clock::rep stop() {
		assert(m_Start.has_value());
		auto time = (steady_clock::now() - *m_Start).count();
		m_Times[m_CurrentIndex] = time;
		m_CurrentIndex = (m_CurrentIndex + 1) % m_Times.size();
		m_Start.reset();
		if (m_Count < m_Times.size()) {
			m_Count += 1;
		}
		return duration_cast<Duration>(steady_clock::duration(time)).count();
	};
	steady_clock::rep get() {
		auto value = std::accumulate(m_Times.begin(), m_Times.end(), 0) / m_Count;
		return duration_cast<Duration>(steady_clock::duration(value)).count();
	}
};

inline const char * vkResultString(vk::Result result) {
	using enum vk::Result;
	switch (result) {
		case eSuccess: return "eSuccess"; 
		case eNotReady: return "eNotReady"; 
		case eTimeout: return "eTimeout"; 
		case eEventSet: return "eEventSet"; 
		case eEventReset: return "eEventReset"; 
		case eIncomplete: return "eIncomplete"; 
		case eErrorOutOfHostMemory: return "eErrorOutOfHostMemory"; 
		case eErrorOutOfDeviceMemory: return "eErrorOutOfDeviceMemory"; 
		case eErrorInitializationFailed: return "eErrorInitializationFailed"; 
		case eErrorDeviceLost: return "eErrorDeviceLost"; 
		case eErrorMemoryMapFailed: return "eErrorMemoryMapFailed"; 
		case eErrorLayerNotPresent: return "eErrorLayerNotPresent"; 
		case eErrorExtensionNotPresent: return "eErrorExtensionNotPresent"; 
		case eErrorFeatureNotPresent: return "eErrorFeatureNotPresent"; 
		case eErrorIncompatibleDriver: return "eErrorIncompatibleDriver"; 
		case eErrorTooManyObjects: return "eErrorTooManyObjects"; 
		case eErrorFormatNotSupported: return "eErrorFormatNotSupported"; 
		case eErrorFragmentedPool: return "eErrorFragmentedPool"; 
		case eErrorUnknown: return "eErrorUnknown"; 
		case eErrorValidationFailed: return "eErrorValidationFailed";
		case eErrorOutOfPoolMemory: return "eErrorOutOfPoolMemory"; 
		case eErrorInvalidExternalHandle: return "eErrorInvalidExternalHandle"; 
		case eErrorFragmentation: return "eErrorFragmentation"; 
		case eErrorInvalidOpaqueCaptureAddress: return "eErrorInvalidOpaqueCaptureAddress"; 
		case ePipelineCompileRequired: return "ePipelineCompileRequired"; 
		case eErrorNotPermitted: return "eErrorNotPermitted"; 
		case eErrorSurfaceLostKHR: return "eErrorSurfaceLostKHR"; 
		case eErrorNativeWindowInUseKHR: return "eErrorNativeWindowInUseKHR"; 
		case eSuboptimalKHR: return "eSuboptimalKHR"; 
		case eErrorOutOfDateKHR: return "eErrorOutOfDateKHR"; 
		case eErrorIncompatibleDisplayKHR: return "eErrorIncompatibleDisplayKHR"; 
		case eErrorInvalidShaderNV: return "eErrorInvalidShaderNV"; 
		case eErrorImageUsageNotSupportedKHR: return "eErrorImageUsageNotSupportedKHR"; 
		case eErrorVideoPictureLayoutNotSupportedKHR: return "eErrorVideoPictureLayoutNotSupportedKHR"; 
		case eErrorVideoProfileOperationNotSupportedKHR: return "eErrorVideoProfileOperationNotSupportedKHR"; 
		case eErrorVideoProfileFormatNotSupportedKHR: return "eErrorVideoProfileFormatNotSupportedKHR"; 
		case eErrorVideoProfileCodecNotSupportedKHR: return "eErrorVideoProfileCodecNotSupportedKHR"; 
		case eErrorVideoStdVersionNotSupportedKHR: return "eErrorVideoStdVersionNotSupportedKHR"; 
		case eErrorInvalidDrmFormatModifierPlaneLayoutEXT: return "eErrorInvalidDrmFormatModifierPlaneLayoutEXT"; 
		#if defined( VK_USE_PLATFORM_WIN32_KHR )
		case eErrorFullScreenExclusiveModeLostEXT: return "eErrorFullScreenExclusiveModeLostEXT";
		#endif /*VK_USE_PLATFORM_WIN32_KHR*/
		case eThreadIdleKHR: return "eThreadIdleKHR";
		case eThreadDoneKHR: return "eThreadDoneKHR";
		case eOperationDeferredKHR: return "eOperationDeferredKHR";
		case eOperationNotDeferredKHR: return "eOperationNotDeferredKHR";
		case eErrorInvalidVideoStdParametersKHR: return "eErrorInvalidVideoStdParametersKHR";
		case eErrorCompressionExhaustedEXT: return "eErrorCompressionExhaustedEXT";
		case eIncompatibleShaderBinaryEXT: return "eIncompatibleShaderBinaryEXT";
		case ePipelineBinaryMissingKHR: return "ePipelineBinaryMissingKHR";
		case eErrorNotEnoughSpaceKHR: return "eErrorNotEnoughSpaceKHR";
		case eErrorPresentTimingQueueFullEXT: return "eErrorPresentTimingQueueFullEXT";
		default: return "Unknown vk::Result value";
	}
	
}

// void unique_println(std::string label, std::string value) {
// 	static std::unordered_map<std::string, std::string> labels;
// 	if (labels.contains(label)) {
// 		if (labels.at(label) != value) {
// 			std::println("{}: {}", label, value);
// 			labels.at(label) = value;
// 		}
// 	} else {
// 		std::println("{}: {}", label, value);
// 		labels.at(label) = value;
// 	}
// }
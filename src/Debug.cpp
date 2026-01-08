#include "Debug.h"
#include "Util.hpp"
#include "Context.h"

DebugNameState g_DebugNameState;
DebugFrameStats g_DebugFrameStats;
Settings g_Settings;

void DebugNameState::SetDebugName(vk::DebugUtilsObjectNameInfoEXT &nameInfo) const {
    Unwrap(vkc->device.setDebugUtilsObjectNameEXT(&nameInfo, vkc->dldid), "Buffer debug naming failed");
}
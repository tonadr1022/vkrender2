#ifdef __cplusplus

#define TUINT32 uint32_t
#define TUINT64 uint64_t

#define TDECL_SHARED_PUSH_CONSTANT(name) \
    struct name
#else

#define TDECL_SHARED_PUSH_CONSTANT(name) \
    layout (push_constant, scalar) uniform name
#endif // __cplusplus

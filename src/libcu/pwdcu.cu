#include <stddefcu.h>
#include <pwdcu.h>

__BEGIN_DECLS;

/* search user database for a name */
__device__ struct passwd *getpwnam_(const char *name)
{
	panic("Not Implemented");
	return nullptr;
}

/* search user database for a user ID */
__device__ struct passwd *getpwuid_(uid_t uid)
{
	panic("Not Implemented");
	return nullptr;
}

/* close the user database */
__device__ void endpwent_()
{
	panic("Not Implemented");
}

/* get user database entry */
__device__ struct passwd *getpwent_()
{
	panic("Not Implemented");
	return nullptr;
}

/* reset user database to first entry */
__device__ void setpwent_()
{
	panic("Not Implemented");
}

__END_DECLS;

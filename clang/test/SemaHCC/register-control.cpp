// RUN: %clang_cc1 -std=c++amp -x hc-kernel -triple amdgcn %s -verify

typedef struct {int x,y,z;} grid_launch_parm;
int x = 1;
const int y = 1;

__attribute__((hc_grid_launch))
void kernel1(grid_launch_parm glp, int *x)
[[hc_waves_per_eu]] // expected-error{{'hc_waves_per_eu' attribute takes at least 1 argument}}
[[hc_waves_per_eu()]] // expected-error{{'hc_waves_per_eu' attribute takes at least 1 argument}}
[[hc_waves_per_eu(x)]] // expected-error{{'hc_waves_per_eu' attribute requires an integer constant}}
[[hc_waves_per_eu(y)]]
[[hc_waves_per_eu(0.5)]] // expected-error{{'hc_waves_per_eu' attribute requires an integer constant}}
[[hc_waves_per_eu(1)]]
[[hc_waves_per_eu(0)]]
[[hc_waves_per_eu(-1)]]
[[hc_waves_per_eu(1,2)]]
[[hc_waves_per_eu(2,1)]] // expected-error{{'hc_waves_per_eu' attribute argument is invalid: min must not be greater than max}}
[[hc_waves_per_eu(0,2)]] // expected-error{{'hc_waves_per_eu' attribute argument is invalid: max must be 0 since min is 0}}
[[hc_waves_per_eu(1,0)]]
[[hc_waves_per_eu(0,0)]]
[[hc_waves_per_eu(-2,-1)]]
[[hc_waves_per_eu(1,-2)]]
[[hc_waves_per_eu(1,"gfx803")]] // expected-error{{'hc_waves_per_eu' attribute requires an integer constant}}
[[hc_waves_per_eu(1,2,"fiji")]] // expected-error{{invalid AMD GPU ISA version parameter 'fiji'}}
[[hc_waves_per_eu(1,2,gfx803)]] // expected-error{{use of undeclared identifier 'gfx803'}}
[[hc_waves_per_eu(1,2,x)]] // expected-error{{'hc_waves_per_eu' attribute requires a string}}

[[hc_flat_workgroup_size]] // expected-error{{'hc_flat_workgroup_size' attribute takes at least 1 argument}}
[[hc_flat_workgroup_size()]] // expected-error{{'hc_flat_workgroup_size' attribute takes at least 1 argument}}
[[hc_flat_workgroup_size(x)]] // expected-error{{'hc_flat_workgroup_size' attribute requires an integer constant}}
[[hc_flat_workgroup_size(y)]]
[[hc_flat_workgroup_size(0.5)]] // expected-error{{'hc_flat_workgroup_size' attribute requires an integer constant}}
[[hc_flat_workgroup_size(1)]]
[[hc_flat_workgroup_size(0)]]
[[hc_flat_workgroup_size(-1)]]
[[hc_flat_workgroup_size(1,2)]]
[[hc_flat_workgroup_size(0,2)]] // expected-error{{'hc_flat_workgroup_size' attribute argument is invalid: max must be 0 since min is 0}}
[[hc_flat_workgroup_size(2,0)]] // expected-error{{'hc_flat_workgroup_size' attribute argument is invalid: min must not be greater than max}}
[[hc_flat_workgroup_size(0,0)]]
[[hc_flat_workgroup_size(-2,-1)]]
[[hc_flat_workgroup_size(2,1)]] // expected-error{{'hc_flat_workgroup_size' attribute argument is invalid: min must not be greater than max}}
[[hc_flat_workgroup_size(1,"gfx803")]] // expected-error{{'hc_flat_workgroup_size' attribute requires an integer constant}}
[[hc_flat_workgroup_size(1,2,"fiji")]] // expected-error{{invalid AMD GPU ISA version parameter 'fiji'}}
[[hc_flat_workgroup_size(1,2,gfx803)]] // expected-error{{use of undeclared identifier 'gfx803'}}
[[hc_flat_workgroup_size(1,2,x)]] // expected-error{{'hc_flat_workgroup_size' attribute requires a string}}

[[hc_max_workgroup_dim]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 1 argument}}
[[hc_max_workgroup_dim()]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 1 argument}}
[[hc_max_workgroup_dim(1)]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 3 arguments}}
[[hc_max_workgroup_dim(1,2)]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 3 arguments}}
[[hc_max_workgroup_dim(x,1,1)]] // expected-error{{'hc_max_workgroup_dim' attribute requires an integer constant}}
[[hc_max_workgroup_dim(y,1,1)]]
[[hc_max_workgroup_dim(0.5,1,1)]] // expected-error{{'hc_max_workgroup_dim' attribute requires an integer constant}}
[[hc_max_workgroup_dim(3,2,1)]]
[[hc_max_workgroup_dim(0,1,1)]]
[[hc_max_workgroup_dim(1,0,1)]]
[[hc_max_workgroup_dim(1,1,0)]]
[[hc_max_workgroup_dim(-1,1,2)]]
[[hc_max_workgroup_dim(10000000000000,1,2)]] // expected-error{{integer constant expression evaluates to value 10000000000000 that cannot be represented in a 32-bit unsigned integer type}}
[[hc_max_workgroup_dim(1,"gfx803")]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 3 arguments}}
[[hc_max_workgroup_dim(1,2,"gfx803")]] // expected-error{{'hc_max_workgroup_dim' attribute requires an integer constant}}
[[hc_max_workgroup_dim(1,2,3,"gfx803")]]
[[hc_max_workgroup_dim(1,2,3,"fiji")]] // expected-error{{invalid AMD GPU ISA version parameter 'fiji'}}
[[hc_max_workgroup_dim(1,2,3,gfx803)]] // expected-error{{use of undeclared identifier 'gfx803'}}
[[hc_max_workgroup_dim(1,2,3,x)]] // expected-error{{'hc_max_workgroup_dim' attribute requires a string}}
{
  x[0] = 0;
}

class A {
  void operator()(int *x)
  [[hc]]
  [[hc_waves_per_eu]] // expected-error{{'hc_waves_per_eu' attribute takes at least 1 argument}}
  [[hc_waves_per_eu()]] // expected-error{{'hc_waves_per_eu' attribute takes at least 1 argument}}
  [[hc_waves_per_eu(x)]] // expected-error{{'hc_waves_per_eu' attribute requires an integer constant}}
  [[hc_waves_per_eu(y)]]
  [[hc_waves_per_eu(0.5)]] // expected-error{{'hc_waves_per_eu' attribute requires an integer constant}}
  [[hc_waves_per_eu(1)]]
  [[hc_waves_per_eu(0)]]
  [[hc_waves_per_eu(-1)]]
  [[hc_waves_per_eu(1,2)]]
  [[hc_waves_per_eu(2,1)]] // expected-error{{'hc_waves_per_eu' attribute argument is invalid: min must not be greater than max}}
  [[hc_waves_per_eu(0,2)]] // expected-error{{'hc_waves_per_eu' attribute argument is invalid: max must be 0 since min is 0}}
  [[hc_waves_per_eu(1,0)]]
  [[hc_waves_per_eu(0,0)]]
  [[hc_waves_per_eu(-2,-1)]]
  [[hc_waves_per_eu(1,-2)]]
  [[hc_waves_per_eu(1,"gfx803")]] // expected-error{{'hc_waves_per_eu' attribute requires an integer constant}}
  [[hc_waves_per_eu(1,2,"fiji")]] // expected-error{{invalid AMD GPU ISA version parameter 'fiji'}}
  [[hc_waves_per_eu(1,2,gfx803)]] // expected-error{{use of undeclared identifier 'gfx803'}}
  [[hc_waves_per_eu(1,2,x)]] // expected-error{{'hc_waves_per_eu' attribute requires a string}}

  [[hc_flat_workgroup_size]] // expected-error{{'hc_flat_workgroup_size' attribute takes at least 1 argument}}
  [[hc_flat_workgroup_size()]] // expected-error{{'hc_flat_workgroup_size' attribute takes at least 1 argument}}
  [[hc_flat_workgroup_size(x)]] // expected-error{{'hc_flat_workgroup_size' attribute requires an integer constant}}
  [[hc_flat_workgroup_size(y)]]
  [[hc_flat_workgroup_size(0.5)]] // expected-error{{'hc_flat_workgroup_size' attribute requires an integer constant}}
  [[hc_flat_workgroup_size(1)]]
  [[hc_flat_workgroup_size(0)]]
  [[hc_flat_workgroup_size(-1)]]
  [[hc_flat_workgroup_size(1,2)]]
  [[hc_flat_workgroup_size(0,2)]] // expected-error{{'hc_flat_workgroup_size' attribute argument is invalid: max must be 0 since min is 0}}
  [[hc_flat_workgroup_size(2,0)]] // expected-error{{'hc_flat_workgroup_size' attribute argument is invalid: min must not be greater than max}}
  [[hc_flat_workgroup_size(0,0)]]
  [[hc_flat_workgroup_size(-2,-1)]]
  [[hc_flat_workgroup_size(2,1)]] // expected-error{{'hc_flat_workgroup_size' attribute argument is invalid: min must not be greater than max}}
  [[hc_flat_workgroup_size(1,"gfx803")]] // expected-error{{'hc_flat_workgroup_size' attribute requires an integer constant}}
  [[hc_flat_workgroup_size(1,2,"fiji")]] // expected-error{{invalid AMD GPU ISA version parameter 'fiji'}}
  [[hc_flat_workgroup_size(1,2,gfx803)]] // expected-error{{use of undeclared identifier 'gfx803'}}
  [[hc_flat_workgroup_size(1,2,x)]] // expected-error{{'hc_flat_workgroup_size' attribute requires a string}}

  [[hc_max_workgroup_dim]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 1 argument}}
  [[hc_max_workgroup_dim()]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 1 argument}}
  [[hc_max_workgroup_dim(1)]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 3 arguments}}
  [[hc_max_workgroup_dim(1,2)]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 3 arguments}}
  [[hc_max_workgroup_dim(x,1,1)]] // expected-error{{'hc_max_workgroup_dim' attribute requires an integer constant}}
  [[hc_max_workgroup_dim(y,1,1)]]
  [[hc_max_workgroup_dim(0.5,1,1)]] // expected-error{{'hc_max_workgroup_dim' attribute requires an integer constant}}
  [[hc_max_workgroup_dim(3,2,1)]]
  [[hc_max_workgroup_dim(0,1,1)]]
  [[hc_max_workgroup_dim(1,0,1)]]
  [[hc_max_workgroup_dim(1,1,0)]]
  [[hc_max_workgroup_dim(-1,1,2)]]
  [[hc_max_workgroup_dim(10000000000000,1,2)]] // expected-error{{integer constant expression evaluates to value 10000000000000 that cannot be represented in a 32-bit unsigned integer type}}
  [[hc_max_workgroup_dim(1,"gfx803")]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 3 arguments}}
  [[hc_max_workgroup_dim(1,2,"gfx803")]] // expected-error{{'hc_max_workgroup_dim' attribute requires an integer constant}}
  [[hc_max_workgroup_dim(1,2,3,"gfx803")]]
  [[hc_max_workgroup_dim(1,2,3,"fiji")]] // expected-error{{invalid AMD GPU ISA version parameter 'fiji'}}
  [[hc_max_workgroup_dim(1,2,3,gfx803)]] // expected-error{{use of undeclared identifier 'gfx803'}}
  [[hc_max_workgroup_dim(1,2,3,x)]] // expected-error{{'hc_max_workgroup_dim' attribute requires a string}}
    {
    x[0] = 123;
    }
};

template<typename T> void foo(T);
int f(unsigned x) {
  //unsigned y = 1;
  foo([&](void)
      [[hc]]
      [[hc_waves_per_eu]] // expected-error{{'hc_waves_per_eu' attribute takes at least 1 argument}}
      [[hc_waves_per_eu()]] // expected-error{{'hc_waves_per_eu' attribute takes at least 1 argument}}
      [[hc_waves_per_eu(x)]] // expected-error{{'hc_waves_per_eu' attribute requires an integer constant}}
      [[hc_waves_per_eu(y)]]
      [[hc_waves_per_eu(0.5)]] // expected-error{{'hc_waves_per_eu' attribute requires an integer constant}}
      [[hc_waves_per_eu(1)]]
      [[hc_waves_per_eu(0)]]
      [[hc_waves_per_eu(-1)]]
      [[hc_waves_per_eu(1,2)]]
      [[hc_waves_per_eu(2,1)]] // expected-error{{'hc_waves_per_eu' attribute argument is invalid: min must not be greater than max}}
      [[hc_waves_per_eu(0,2)]] // expected-error{{'hc_waves_per_eu' attribute argument is invalid: max must be 0 since min is 0}}
      [[hc_waves_per_eu(1,0)]]
      [[hc_waves_per_eu(0,0)]]
      [[hc_waves_per_eu(-2,-1)]]
      [[hc_waves_per_eu(1,-2)]]
      [[hc_waves_per_eu(1,"gfx803")]] // expected-error{{'hc_waves_per_eu' attribute requires an integer constant}}
      [[hc_waves_per_eu(1,2,"fiji")]] // expected-error{{invalid AMD GPU ISA version parameter 'fiji'}}
      [[hc_waves_per_eu(1,2,gfx803)]] // expected-error{{use of undeclared identifier 'gfx803'}}
      [[hc_waves_per_eu(1,2,x)]] // expected-error{{'hc_waves_per_eu' attribute requires a string}}

      [[hc_flat_workgroup_size]] // expected-error{{'hc_flat_workgroup_size' attribute takes at least 1 argument}}
      [[hc_flat_workgroup_size()]] // expected-error{{'hc_flat_workgroup_size' attribute takes at least 1 argument}}
      [[hc_flat_workgroup_size(x)]] // expected-error{{'hc_flat_workgroup_size' attribute requires an integer constant}}
      [[hc_flat_workgroup_size(y)]]
      [[hc_flat_workgroup_size(0.5)]] // expected-error{{'hc_flat_workgroup_size' attribute requires an integer constant}}
      [[hc_flat_workgroup_size(1)]]
      [[hc_flat_workgroup_size(0)]]
      [[hc_flat_workgroup_size(-1)]]
      [[hc_flat_workgroup_size(1,2)]]
      [[hc_flat_workgroup_size(0,2)]] // expected-error{{'hc_flat_workgroup_size' attribute argument is invalid: max must be 0 since min is 0}}
      [[hc_flat_workgroup_size(2,0)]] // expected-error{{'hc_flat_workgroup_size' attribute argument is invalid: min must not be greater than max}}
      [[hc_flat_workgroup_size(0,0)]]
      [[hc_flat_workgroup_size(-2,-1)]]
      [[hc_flat_workgroup_size(2,1)]] // expected-error{{'hc_flat_workgroup_size' attribute argument is invalid: min must not be greater than max}}
      [[hc_flat_workgroup_size(1,"gfx803")]] // expected-error{{'hc_flat_workgroup_size' attribute requires an integer constant}}
      [[hc_flat_workgroup_size(1,2,"fiji")]] // expected-error{{invalid AMD GPU ISA version parameter 'fiji'}}
      [[hc_flat_workgroup_size(1,2,gfx803)]] // expected-error{{use of undeclared identifier 'gfx803'}}
      [[hc_flat_workgroup_size(1,2,x)]] // expected-error{{'hc_flat_workgroup_size' attribute requires a string}}

      [[hc_max_workgroup_dim]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 1 argument}}
      [[hc_max_workgroup_dim()]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 1 argument}}
      [[hc_max_workgroup_dim(1)]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 3 arguments}}
      [[hc_max_workgroup_dim(1,2)]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 3 arguments}}
      [[hc_max_workgroup_dim(x,1,1)]] // expected-error{{'hc_max_workgroup_dim' attribute requires an integer constant}}
      [[hc_max_workgroup_dim(y,1,1)]]
      [[hc_max_workgroup_dim(0.5,1,1)]] // expected-error{{'hc_max_workgroup_dim' attribute requires an integer constant}}
      [[hc_max_workgroup_dim(3,2,1)]]
      [[hc_max_workgroup_dim(0,1,1)]]
      [[hc_max_workgroup_dim(1,0,1)]]
      [[hc_max_workgroup_dim(1,1,0)]]
      [[hc_max_workgroup_dim(-1,1,2)]]
      [[hc_max_workgroup_dim(10000000000000,1,2)]] // expected-error{{integer constant expression evaluates to value 10000000000000 that cannot be represented in a 32-bit unsigned integer type}}
      [[hc_max_workgroup_dim(1,"gfx803")]] // expected-error{{'hc_max_workgroup_dim' attribute takes at least 3 arguments}}
      [[hc_max_workgroup_dim(1,2,"gfx803")]] // expected-error{{'hc_max_workgroup_dim' attribute requires an integer constant}}
      [[hc_max_workgroup_dim(1,2,3,"gfx803")]]
      [[hc_max_workgroup_dim(1,2,3,"fiji")]] // expected-error{{invalid AMD GPU ISA version parameter 'fiji'}}
      [[hc_max_workgroup_dim(1,2,3,gfx803)]] // expected-error{{use of undeclared identifier 'gfx803'}}
      [[hc_max_workgroup_dim(1,2,3,x)]] // expected-error{{'hc_max_workgroup_dim' attribute requires a string}}
      {});

  return 0;
}


#include "pch.h"

#include <Common\Math.h>
#include <glm\gtx\compatibility.hpp>

// Lazy copy-past code here
// TODO do not copy paste the code but let the unit test access the hidden part of the RayTracer project and use the cu/cuh files directly.
namespace rt
{

// Core of raytracing
class Ray
{
public:
  Ray( const math::vec3& o = math::vec3( 0.0f )
                           , const math::vec3& d = math::vec3( 0.0f )
                           , const bool normalizeDirection = true )
    : mOrigin( o )
    , mDirection( normalizeDirection ? glm::normalize( d ) : d )
  {}

  Ray( const Ray& rhs )
    : mOrigin( rhs.mOrigin )
    , mDirection( rhs.mDirection )
  {}

  Ray& operator = ( const Ray& rhs )
  {
    mOrigin = rhs.mOrigin;
    mDirection = rhs.mDirection;
    return *this;
  }

  const math::vec3& origin() const
  {
    return mOrigin;
  }

  const math::vec3& direction() const
  {
    return mDirection;
  }

  math::vec3 point( const float t ) const
  {
    return mOrigin + mDirection * t;
  }

private:
  math::vec3 mOrigin;
  math::vec3 mDirection;
};

// tests of a given ray hits the given triangle <v0, v1, v2>
// return true if hit occours and sets t, u, v values
// if false is returned, t, u, v meaningless
bool Hit( const Ray& ray
          , const math::vec3& v0
          , const math::vec3& v1
          , const math::vec3& v2
          , float& t
          , float& u
          , float& v )
{
  const math::vec3 v0v1( v1 - v0 );
  const math::vec3 v0v2( v2 - v0 );
  const math::vec3 pv( glm::cross( ray.direction(), v0v2 ) );
  const float det( glm::dot( v0v1, pv ) );

  if ( det < glm::epsilon<float>() )
  {
    return false; // backfacing triangle not visible (Culling)
  }

  const float invDet( 1.0f / det );

  const math::vec3 tv( ray.origin() - v0 );
  u = glm::dot( tv, pv ) * invDet;
  if ( u < 0.0f || u > 1.0f )
  {
    return false;
  }

  const math::vec3 qv( glm::cross( tv, v0v1 ) );
  v = glm::dot( ray.direction(), qv ) * invDet;
  if ( v < 0.0f || u + v > 1.0f )
  {
    return false;
  }

  t = glm::dot( v0v2, qv ) * invDet;
  return true;
}

// Mesh storage
// mesh data on GPU:
// chunk of float array for vertex coordinates
// chunk of uint64_t array for triangle vertex indices ( indexing into the vertex coord array )
class Triangle
{
public:
  Triangle( const uint64_t* data )
    : mDataPtr( data )
  {}

  const uint64_t& a() const
  {
    return *mDataPtr;
  }

  const uint64_t& b() const
  {
    return *( mDataPtr + 1 );
  }

  const uint64_t& c() const
  {
    return *( mDataPtr + 2 );
  }

private:
  const uint64_t* mDataPtr; // TODO use mData, mData+1, mData+2 as a,b,c but nicer than this
};

// create only on device and add ability to get data from host
// initialize with a big chunk of memory, follow xxx|yyy|zzz order instead of xyzxyzxyz
//
// TODO validate if triangle indices are inside the vertex pool
// think a nice vertexpool + users architecture
class VertexPool
{
public:
  // data layout:
  // [x0,y0,z0][x1,y1,z1]..[xN,yN,zN]
  VertexPool( const float* data )
    : mDataPtr( data )
  {}

  bool triangle( const Triangle& triangle, math::vec3& a, math::vec3& b, math::vec3& c ) const
  {
    a = math::vec3( x( triangle.a() ), y( triangle.a() ), z( triangle.a() ) );
    b = math::vec3( x( triangle.b() ), y( triangle.b() ), z( triangle.b() ) );
    c = math::vec3( x( triangle.c() ), y( triangle.c() ), z( triangle.c() ) );
    return true;
  }

private:
  const float& x( const uint64_t& vertexIdx ) const
  {
    const size_t vertexOffset( ( vertexIdx * 3 ) );
    return *( mDataPtr + vertexOffset );
  }

  const float& y( const uint64_t& vertexIdx ) const
  {
    const size_t vertexOffset( ( vertexIdx * 3 ) );
    return *( mDataPtr + vertexOffset + 1 );
  }

  const float& z( const uint64_t& vertexIdx ) const
  {
    const size_t vertexOffset( ( vertexIdx * 3 ) );
    return *( mDataPtr + vertexOffset + 2 );
  }

private:
  const float* mDataPtr;
};

}

// ray from 0,0,0 towards 0,0,-1 hits the triangle at z=-10 with hit point 0,0,-10
TEST( TestCaseName, hit_at_vtx_a )
{
  const rt::Ray r( math::vec3( 0.0f, 0.0f, 0.0f ), math::vec3( 0.0f, 0.0f, -1.0f ), true );

  // CCW triangle
  const float vertices[] = { 0.0, 0.0, -10.0,   1.0, 0.0, -10.0,   0.0, 1.0, -10.0 };
  const rt::VertexPool vp( vertices );

  const uint64_t indices[] = { 0, 1, 2 };
  const rt::Triangle tri( indices );

  math::vec3 a, b, c;
  ASSERT_TRUE( vp.triangle( tri, a, b, c ) );

  float t( 0.0f ), u( 0.0f ), v( 0.0f );
  ASSERT_TRUE( Hit( r, a, b, c, t, u, v ) );

  const math::vec3 n( glm::normalize( glm::cross( b - a, c - a ) ) );
  ASSERT_EQ( n, math::vec3( 0.0f, 0.0f, 1.0f ) );

  const math::vec3 hitpoint( r.point( t ) );
  EXPECT_EQ( hitpoint, math::vec3( 0.0f, 0.0f, -10.0f ) );
}

// ray from 0,0,0 towards 0,0,-1 hits the triangle at z=-10 with hit point 0,0,-10
TEST( TestCaseName, hit_triangle_edge )
{
  const rt::Ray r( math::vec3( 0.5f, 0.5f, 0.0f ), math::vec3( 0.0f, 0.0f, -1.0f ), true );

  // CCW triangle
  const float vertices[] = { 0.0, 0.0, -10.0,   1.0, 0.0, -10.0,   0.0, 1.0, -10.0 };
  const rt::VertexPool vp( vertices );

  const uint64_t indices[] = { 0, 1, 2 };
  const rt::Triangle tri( indices );

  math::vec3 a, b, c;
  ASSERT_TRUE( vp.triangle( tri, a, b, c ) );

  float t( 0.0f ), u( 0.0f ), v( 0.0f );
  ASSERT_TRUE( Hit( r, a, b, c, t, u, v ) );

  const math::vec3 n( glm::normalize( glm::cross( b - a, c - a ) ) );
  ASSERT_EQ( n, math::vec3( 0.0f, 0.0f, 1.0f ) );

  const math::vec3 hitpoint( r.point( t ) );
  EXPECT_EQ( hitpoint, math::vec3( 0.5f, 0.5f, -10.0f ) );
}

// ray from 0,0,0 towards 0,0,1 misses the triangle at z=-10 as it goes to the oppisite direction (culling)
TEST( TestCaseName, miss_opposite_direction )
{
  const rt::Ray r( math::vec3( 0.0f, 0.0f, 0.0f ), math::vec3( 0.0f, 0.0f, 1.0f ), true );

  // CCW triangle
  const float vertices[] = { 0.0, 0.0, -10.0,   1.0, 0.0, -10.0,   0.0, 1.0, -10.0 };
  const rt::VertexPool vp( vertices );

  const uint64_t indices[] = { 0, 1, 2 };
  const rt::Triangle tri( indices );

  math::vec3 a, b, c;
  ASSERT_TRUE( vp.triangle( tri, a, b, c ) );

  float t( 0.0f ), u( 0.0f ), v( 0.0f );
  EXPECT_FALSE( Hit( r, a, b, c, t, u, v ) );
}

// ray from 1,1,0 towards 0,0,-1 hits triangle's plane but misses the triangle
TEST( TestCaseName, miss_out_of_triangle )
{
  const rt::Ray r( math::vec3( 1.0f, 1.0f, 0.0f ), math::vec3( 0.0f, 0.0f, 1.0f ), true );

  // CCW triangle
  const float vertices[] = { 0.0, 0.0, -10.0,   1.0, 0.0, -10.0,   0.0, 1.0, -10.0 };
  const rt::VertexPool vp( vertices );

  const uint64_t indices[] = { 0, 1, 2 };
  const rt::Triangle tri( indices );

  math::vec3 a, b, c;
  ASSERT_TRUE( vp.triangle( tri, a, b, c ) );

  float t( 0.0f ), u( 0.0f ), v( 0.0f );
  EXPECT_FALSE( Hit( r, a, b, c, t, u, v ) );
}

// ray from close to origin towards 0,0,-1 hits triangle's plane but misses the triangle
TEST( TestCaseName, miss_vtx_a_below_x )
{
  const rt::Ray r( math::vec3( -glm::epsilon<float>(), 0.0f, 0.0f ), math::vec3( 0.0f, 0.0f, -1.0f ), true ); // TODO double check the epsilon

  // CCW triangle
  const float vertices[] = { 0.0, 0.0, -10.0,   1.0, 0.0, -10.0,   0.0, 1.0, -10.0 };
  const rt::VertexPool vp( vertices );

  const uint64_t indices[] = { 0, 1, 2 };
  const rt::Triangle tri( indices );

  math::vec3 a, b, c;
  ASSERT_TRUE( vp.triangle( tri, a, b, c ) );

  float t( 0.0f ), u( 0.0f ), v( 0.0f );
  EXPECT_FALSE( Hit( r, a, b, c, t, u, v ) );
}

// ray from close to origin towards 0,0,-1 hits triangle's plane but misses the triangle
TEST( TestCaseName, miss_vtx_a_below_y )
{
  const rt::Ray r( math::vec3( 0.0f, -glm::epsilon<float>(), 0.0f ), math::vec3( 0.0f, 0.0f, -1.0f ), true ); // TODO double check the epsilon

  // CCW triangle
  const float vertices[] = { 0.0, 0.0, -10.0,   1.0, 0.0, -10.0,   0.0, 1.0, -10.0 };
  const rt::VertexPool vp( vertices );

  const uint64_t indices[] = { 0, 1, 2 };
  const rt::Triangle tri( indices );

  math::vec3 a, b, c;
  ASSERT_TRUE( vp.triangle( tri, a, b, c ) );

  float t( 0.0f ), u( 0.0f ), v( 0.0f );
  EXPECT_FALSE( Hit( r, a, b, c, t, u, v ) );
}

// 
TEST( TestCaseName, miss_towards_positive_z )
{
  const rt::Ray r( math::vec3( 0.0f, 0.0f, 0.0f ), math::vec3( 0.0f, 0.0f, 1.0f ), true );

  // CCW triangle
  const float vertices[] = { 0.0, 0.0, 10.0,   1.0, 0.0, 10.0,   0.0, 1.0, 10.0 };
  const rt::VertexPool vp( vertices );

  const uint64_t indices[] = { 0, 1, 2 };
  const rt::Triangle tri( indices );

  math::vec3 a, b, c;
  ASSERT_TRUE( vp.triangle( tri, a, b, c ) );

  float t( 0.0f ), u( 0.0f ), v( 0.0f );
  EXPECT_FALSE( Hit( r, a, b, c, t, u, v ) );
}

// 
TEST( TestCaseName, hit_towards_positive_z )
{
  const rt::Ray r( math::vec3( 0.5f, 0.5f, 0.5f ), math::vec3( 0.0f, 0.0f, 1.0f ), true );

  // CW triangle
  const float vertices[] = { 0.0, 0.0, 10.0,   1.0, 0.0, 10.0,   0.0, 1.0, 10.0 };
  const rt::VertexPool vp( vertices );

  const uint64_t indices[] = { 0, 2, 1 };
  const rt::Triangle tri( indices );

  math::vec3 a, b, c;
  ASSERT_TRUE( vp.triangle( tri, a, b, c ) );

  float t( 0.0f ), u( 0.0f ), v( 0.0f );
  EXPECT_TRUE( Hit( r, a, b, c, t, u, v ) );

  const math::vec3 n( glm::normalize( glm::cross( b - a, c - a ) ) );
  ASSERT_EQ( n, math::vec3( 0.0f, 0.0f, -1.0f ) );

  const math::vec3 hitpoint( r.point( t ) );
  EXPECT_EQ( hitpoint, math::vec3( 0.5f, 0.5f, 10.0f ) );
}
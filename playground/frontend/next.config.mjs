/** @type {import('next').NextConfig} */
const nextConfig = {
  // Docker: standalone output for minimal production image
  output: "standalone",
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;

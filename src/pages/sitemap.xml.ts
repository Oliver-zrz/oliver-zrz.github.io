import { posts } from '../lib/posts';
export function GET() {
  const paths = ['/', '/blog/', '/about/', '/toolbox/', ...posts.map(p => p.url)];
  return new Response(`<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${paths.map(path => `<url><loc>https://oliver-zrz.github.io${path}</loc></url>`).join('')}</urlset>`, {headers: {'Content-Type':'application/xml'}});
}

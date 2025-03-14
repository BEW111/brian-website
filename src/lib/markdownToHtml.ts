import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";
import rehypeDocument from "rehype-document";

export default async function markdownToHtml(markdown: string) {
  const result = await unified()
    .use(remarkParse, { fragment: true })
    .use(remarkMath)
    .use(remarkRehype)
    .use(rehypeDocument, {
      // Get the latest one from: <https://katex.org/docs/browser>.
      css: "https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css",
    })
    .use(rehypeKatex)
    .use(rehypeStringify)
    .process(markdown);

  // const result = await remark().use(html).process(markdown);
  return result.toString();
}

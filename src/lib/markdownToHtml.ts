import { unified } from "unified";
import rehypeDocument from "rehype-document";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";

export default async function markdownToHtml(markdown: string) {
  const result = await unified()
    .use(remarkParse, { fragment: true })
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkRehype)
    .use(rehypeDocument, {
      // Get the latest one from: <https://katex.org/docs/browser>.
      css: [
        "https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.0/styles/github-dark.min.css",
      ],
    })
    .use(rehypeKatex)
    .use(rehypeHighlight)
    .use(rehypeStringify)
    .process(markdown);

  return result.toString();
}

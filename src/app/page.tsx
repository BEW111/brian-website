import Container from "@/app/_components/container";
import { Intro } from "@/app/_components/intro";
import { PostPreviewList } from "@/app/_components/post-preview-list";
import { getAllPosts } from "@/lib/api";
import { SectionSeparator } from "./_components/section-separator";

export default function Index() {
  const allPosts = getAllPosts();

  return (
    <main>
      <Container>
        <Intro />
        {/* <p>bewilliams111 [at] gmail [dot] com</p>
        <SectionSeparator /> */}
        <h2 className="mb-4 text-5xl md:text-7xl font-semibold tracking-tighter leading-tight">
          notes & thoughts
        </h2>
        <PostPreviewList posts={allPosts} />
      </Container>
    </main>
  );
}
